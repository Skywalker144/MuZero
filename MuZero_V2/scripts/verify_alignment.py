from pathlib import Path
import argparse
import ast
import os
import subprocess
import sys
import tempfile

import torch


def extract_block(source, signature):
    start = source.index(signature)
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'python'))
from muzero.config import load_config

config = load_config(root / 'configs/baseline', environ={})
train_ast = ast.parse((args.source / 'katago/python/train.py').read_text())
defaults = {}
for node in ast.walk(train_ast):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
        if node.args and isinstance(node.args[0], ast.Constant):
            for keyword in node.keywords:
                if keyword.arg == 'default' and isinstance(keyword.value, ast.Constant):
                    defaults[node.args[0].value] = keyword.value.value
metrics_ast = ast.parse((args.source / 'katago/python/katago/train/metrics_pytorch.py').read_text())
methods = {'loss_policy_player_samplewise', 'loss_policy_opponent_samplewise', 'loss_value_samplewise'}
nodes = []
for node in metrics_ast.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'cross_entropy':
        nodes.append(node)
    if isinstance(node, ast.ClassDef) and node.name == 'Metrics':
        node.body = [method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in methods]
        nodes.append(node)
namespace = {'torch': torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<KataGo loss source>', 'exec'), namespace)
metrics = namespace['Metrics']()
metrics.policy_len, metrics.value_len = 25, 3
torch.manual_seed(71)
policy = torch.randn(8, 4, 25, requires_grad=True)
value = torch.randn(8, 3, requires_grad=True)
targets = torch.randn(8, 4, 25).softmax(-1)
outcomes = torch.randn(8, 3).softmax(-1)
weight = torch.rand(8)
mask = torch.ones(8)
reference_loss = metrics.loss_policy_player_samplewise(policy[:, 0], targets[:, 0], mask, weight).sum()
reference_loss += defaults['-soft-policy-weight-scale'] * metrics.loss_policy_player_samplewise(policy[:, 1], targets[:, 1], mask, weight).sum()
reference_loss += metrics.loss_policy_opponent_samplewise(policy[:, 2], targets[:, 2], mask, weight).sum()
reference_loss += defaults['-soft-policy-weight-scale'] * metrics.loss_policy_opponent_samplewise(policy[:, 3], targets[:, 3], mask, weight).sum()
reference_loss += defaults['-value-loss-scale'] * metrics.loss_value_samplewise(value, outcomes, mask, weight).sum()
scales = torch.tensor([1, config['SOFT_POLICY_LOSS_SCALE'], config['OPPONENT_POLICY_LOSS_SCALE'],
                       config['SOFT_POLICY_LOSS_SCALE'] * config['OPPONENT_POLICY_LOSS_SCALE']])
actual_loss = (-(targets * policy.log_softmax(-1)).sum(-1) * scales * weight[:, None]).sum()
actual_loss += config['VALUE_LOSS_SCALE'] * (-(outcomes * value.log_softmax(-1)).sum(-1) * weight).sum()
torch.testing.assert_close(actual_loss, reference_loss)
for actual, expected in zip(torch.autograd.grad(actual_loss, (policy, value), retain_graph=True),
                            torch.autograd.grad(reference_loss, (policy, value))):
    torch.testing.assert_close(actual, expected)
print('KataGo loss source conformance: policy and WDL values and gradients passed', flush=True)
reference = args.source / 'katago/cpp'
helpers = (reference / 'search/searchhelpers.cpp').read_text()
play = (reference / 'program/play.cpp').read_text()
explore = (reference / 'search/searchexplorehelpers.cpp').read_text()
source = r'''
#include "muzero/record.h"
#include <iostream>
#include <cassert>
using namespace std;
namespace reference {
using StringError = runtime_error;
using Player = int;
constexpr int P_WHITE = 1;
void testAssert(bool b) { assert(b); }
struct Search {
    static void computeDirichletAlphaDistribution(int, const float*, double*);
    double getExploreSelectionValueInverse(double, double, double, double, Player) const;
};
struct ValueTargets { double win, loss, noResult; };
struct ReportedSearchValues { double winValue, lossValue, noResultValue; };
struct Reanalysis { bool wasReanalyzed = false; };
struct GameData {
    vector<float> targetWeightByTurn;
    vector<double> policySurpriseByTurn;
    vector<Reanalysis> reanalysisByTurn;
};
struct PlaySettings { double policySurpriseDataWeight, valueSurpriseDataWeight; bool useReanalyze = false; };
'''
source += extract_block(helpers, 'void Search::computeDirichletAlphaDistribution(')
source += extract_block(explore, 'double Search::getExploreSelectionValueInverse(')
source += extract_block(play, 'static double valueSurpriseKL(')
source += extract_block(play, 'static void computeValueSurpriseByTurn(')
source += r'''
vector<float> weights(const muzero::FinishedGame& game, double p, double v) {
    GameData data;
    GameData* gameData = &data;
    PlaySettings playSettings{p,v};
    vector<bool> wasCheapSearchByTurn;
    vector<ValueTargets> searches;
    vector<ReportedSearchValues> networks;
    for (const auto& step : game.steps) {
        data.targetWeightByTurn.push_back(step.weight);
        data.policySurpriseByTurn.push_back(step.policy_surprise);
        data.reanalysisByTurn.emplace_back();
        wasCheapSearchByTurn.push_back(step.weight == 0);
        int win = step.player == 1 ? 0 : 2;
        searches.push_back({step.search_wdl[win],step.search_wdl[2-win],step.search_wdl[1]});
        networks.push_back({step.network_wdl[win],step.network_wdl[2-win],step.network_wdl[1]});
    }
    searches.push_back({double(game.winner==1),double(game.winner==-1),double(game.winner==0)});
    vector<double> valueSurpriseByTurn;
    computeValueSurpriseByTurn(valueSurpriseByTurn,searches,networks,game.size*game.size,false);
'''
source += extract_block(play, 'if(playSettings.policySurpriseDataWeight > 0 || playSettings.valueSurpriseDataWeight > 0)')
source += r'''
    return data.targetWeightByTurn;
}
}
int main(int argc, char** argv) {
    if(argc != 2) return 2;
    muzero::Config raw(argv[1]);
    muzero::SearchConfig c(raw);
    c.use_lcb = false;
    c.policy_target_pruning = true;
    mt19937_64 rng(731);
    uniform_real_distribution<double> uniform(0,1);
    for (int trial=0; trial<500; ++trial) {
        vector<float> policy(25);
        for(auto& p : policy) p=uniform(rng);
        double sum=accumulate(policy.begin(),policy.end(),0.0);
        for(auto& p : policy) p/=sum;
        policy.back()=-1;
        vector<double> alpha(25);
        reference::Search::computeDirichletAlphaDistribution(25,policy.data(),alpha.data());
        auto actual=muzero::dirichlet_alpha_distribution(vector<double>(policy.begin(),policy.end()));
        for(int j=0;j<25;++j) assert(abs(actual[j]-alpha[j])<1e-12);
        muzero::FinishedGame game{0,15,15,muzero::Rule::FREESTYLE,trial%3-1,{},{}};
        for(int j=0;j<20;++j) {
            muzero::Step step{};
            step.player=j%2 ? -1 : 1;
            step.weight=(trial%5 == 0 || uniform(rng)<0.75) ? 0 : 1;
            step.policy_surprise=uniform(rng)*2;
            for(auto* wdl : {&step.search_wdl,&step.network_wdl}) {
                for(auto& x : *wdl) x=uniform(rng);
                double mass=accumulate(wdl->begin(),wdl->end(),0.0);
                for(auto& x : *wdl) x/=mass;
            }
            game.steps.push_back(step);
        }
        double p=trial%2 ? 0.5 : 0, v=trial%3 ? 0.1 : 0;
        auto expected=reference::weights(game,p,v);
        muzero::apply_training_weights(game,p,v);
        for(int j=0;j<20;++j) assert(abs(game.steps[j].weight-expected[j])<1e-6);
        muzero::Node root;
        root.update(0);
        for(int j=0;j<2;++j) {
            auto child=make_unique<muzero::Node>();
            child->action=j;
            child->prior=0.5;
            child->parent=&root;
            int n=j==0 ? 100 : 20+trial%60;
            for(int k=0;k<n;++k) child->update(j==0 ? -0.8 : 0.8);
            root.visits+=n;
            root.children.push_back(move(child));
        }
        double scaling=(c.pb_c_init+log((root.visits+c.pb_c_base+1)/c.pb_c_base))*sqrt(root.visits);
        double best=0.9+scaling*0.5/101;
        double reduced=ceil(min(double(root.children[1]->visits),reference::Search().getExploreSelectionValueInverse(best,scaling,0.5,0.1,1)));
        auto result=muzero::search_result(root,c,2);
        assert(abs(result.move_policy[1]-reduced/(100+reduced))<1e-12);
    }
    cout << "KataGo source conformance: 500 cases of noise, surprise weighting and visit reduction passed\n";
}
'''
with tempfile.TemporaryDirectory(prefix='muzero-alignment-') as tmp:
    directory = Path(tmp)
    path = directory / 'compare.cpp'
    binary = directory / 'compare'
    config = directory / 'resolved.cfg'
    path.write_text(source)
    subprocess.run([os.environ.get('PYTHON', 'python'), str(root / 'python/muzero/config.py'),
                    '--native', '--output', str(config)], check=True)
    generated = directory / 'generated/muzero/protocol.h'
    generated.parent.mkdir(parents=True)
    subprocess.run([os.environ.get('PYTHON', 'python'), str(root / 'scripts/generate_protocol.py'),
                    str(root / 'protocol.json'), str(generated)], check=True)
    subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-pthread',
                    '-I' + str(root / 'cpp/include'), '-I' + str(directory / 'generated'),
                    str(path), str(root / 'cpp/src/search.cpp'), str(root / 'cpp/src/rules.cpp'),
                    str(root / 'cpp/src/training_targets.cpp'), '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(config)], check=True, timeout=30)
