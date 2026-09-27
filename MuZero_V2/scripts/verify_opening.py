from pathlib import Path
import argparse
import json
import os
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--build-dir', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sky = args.source / 'katago/cpp'
build = args.build_dir or root / 'build/core'
source = (sky / 'game/gomokuopening.cpp').read_text()
start = source.index('  Loc getKataGomoRandomNearbyMove(')
end = source.index('  using BatchEvaluator', start)
body = source[start:end]
header = (sky / 'game/gomokuopening.h').read_text()
structs = header[header.index('struct GomokuBalancedOpeningSettings'):header.index('namespace GomokuBalancedOpening')]
constructors = source[source.index('GomokuBalancedOpeningSettings::GomokuBalancedOpeningSettings()'):source.index('namespace {')]
utils = (sky / 'program/playutils.cpp').read_text()
policy = utils[utils.index('Loc PlayUtils::getGameInitializationMove('):utils.index('//Try playing a bunch')]
continuation = utils[utils.index('void PlayUtils::initializeGameUsingPolicyKataGomo('):utils.index('double PlayUtils::getHackedLCB')]
text = '#include ' + json.dumps(str(root / "cpp/src/opening.cpp")) + "\n" + r'''
#include <iostream>
#include <cassert>
#include <sstream>
using namespace std;
namespace reference {
using Loc = int;
using Player = int;
constexpr int C_EMPTY = 0, P_BLACK = 1, P_WHITE = -1;
int getOpp(int p) { return -p; }
using StringError = runtime_error;
void testAssert(bool b) { if (!b) throw runtime_error("Reference assertion"); }
struct Board {
    muzero::Game game;
    int x_size, y_size, stoneCount = 0;
    vector<int> colors;
    static constexpr int NULL_LOC = -1;
    explicit Board(const muzero::Game& g) : game(g), x_size(g.size()), y_size(g.size()), stoneCount(g.turn()), colors(g.board().cells) {}
    bool isEmpty() const { return stoneCount == 0; }
};
namespace Location { int getLoc(int x, int y, int size) { return y*size+x; } }
struct BoardHistory {
    Player presumedNextMovePla = P_BLACK;
    bool isGameFinished = false;
    vector<int> moves;
    bool isLegal(const Board& b, Loc loc, Player p) const {
        return !isGameFinished && p == presumedNextMovePla && loc >= 0 && loc < b.x_size*b.y_size && b.colors[loc] == 0;
    }
    void makeBoardMoveAssumeLegal(Board& b, Loc loc, Player p) {
        testAssert(isLegal(b,loc,p));
        int action = loc / b.x_size * b.game.canvas() + loc % b.x_size;
        b.game.play(action);
        b.colors = b.game.board().cells;
        b.stoneCount = b.game.turn();
        presumedNextMovePla = b.game.player();
        isGameFinished = b.game.finished();
        moves.push_back(action);
    }
};
struct Rand {
    muzero::OpeningRandom random;
    explicit Rand(mt19937_64& engine) : random(engine) {}
    bool nextBool(double p) { return random.coin(p); }
    double nextExponential() { return random.exponential(); }
    double nextGaussianTruncated(double bound) { return random.truncated_gaussian(bound); }
    uint32_t nextUInt(const double* w, size_t n) { return random.choose(vector<double>(w,w+n)); }
    uint32_t nextUInt(uint32_t n) { return random.uniform_index(n); }
};
'''
text += structs + constructors
text += r'''
namespace GomokuBalancedOpening { using PositionEvaluator = function<double(const Board&, const BoardHistory&, Player)>; }
using LegacyEvaluatorSelector = function<void(GomokuBalancedOpening::PositionEvaluator&)>;
enum class LegacyTryStatus { SUCCESS, RETRY, FATAL };
'''
text += body
text += r'''
struct NNOutput { vector<double> policyProbs; int nnXLen, nnYLen; };
struct NNResultBuf { shared_ptr<NNOutput> result; };
struct MiscNNInputParams { double drawEquivalentWinsForWhite; };
struct NNEvaluator {
    muzero::Evaluator& evaluator;
    explicit NNEvaluator(muzero::Evaluator& e) : evaluator(e) {}
    void evaluate(const Board& b, const BoardHistory&, Player p, MiscNNInputParams, NNResultBuf& buf, bool, bool) {
        auto eval = evaluator.initial(b.game.observation(p));
        for (int a=0; a<b.game.actions(); ++a) if (!b.game.legal(a)) eval.logits[a] = -numeric_limits<double>::infinity();
        buf.result = make_shared<NNOutput>();
        buf.result->policyProbs = muzero::softmax(eval.logits);
        buf.result->nnXLen = b.game.canvas();
        buf.result->nnYLen = b.game.canvas();
    }
};
struct Search { NNEvaluator* nnEvaluator; };
namespace NNPos {
int getPolicySize(int x, int y) { return x*y; }
Loc posToLoc(int pos, int sx, int sy, int nx, int) {
    int x=pos%nx, y=pos/nx;
    return x < sx && y < sy ? y*sx+x : -1;
}
}
namespace PlayUtils {
Loc getGameInitializationMove(Search*, Search*, const Board&, const BoardHistory&, Player, NNResultBuf&, Rand&, double);
void initializeGameUsingPolicyKataGomo(Search*, Search*, Board&, BoardHistory&, Player&, Rand&, double, double);
}
'''
text += policy + continuation
text += r'''
}
struct Eval : muzero::Evaluator {
    int calls = 0;
    int mode = 0;
    vector<vector<float>> observations;
    muzero::Evaluation initial(const vector<float>& obs) override {
        ++calls;
        observations.push_back(obs);
        int n=obs.size()/muzero::INPUT_PLANES;
        double sum=0;
        for (size_t i=0; i<obs.size(); ++i) sum += obs[i] * ((i*37)%71-35.0);
        double value = mode == 1 ? .999 : mode == 2 ? 0 : .92*sin(sum*.013);
        if ((mode == 3 && calls == 1) || (mode == 4 && calls == 2) || (mode == 5 && calls == 3))
            value = numeric_limits<double>::quiet_NaN();
        vector<double> logits(n);
        for (int i=0;i<n;++i) logits[i]=sin(i*.81+sum)*2;
        return {nullptr,logits,value};
    }
    muzero::Evaluation recurrent(const shared_ptr<const muzero::Latent>&, int) override { throw runtime_error("dynamics"); }
};
int main(int argc, char** argv) {
    if(argc!=2) return 2;
    int cases=0;
    for(int seed=0;seed<120;++seed) for (auto rule : {muzero::Rule::FREESTYLE,muzero::Rule::STANDARD,muzero::Rule::RENJU}) {
        muzero::OpeningConfig c{muzero::Config(argv[1])};
        c.probability = seed%3 == 0 ? 0 : seed%3 == 1 ? .37 : 1;
        c.policy_init = seed%5 != 0;
        c.policy_after = seed%11 != 0;
        c.policy_on_failure = seed%13 != 0;
        c.policy_init_mean = seed%4 == 0 ? 100 : seed%4 == 1 ? 0 : 6;
        c.rejection_probability = seed%7 == 0 ? 1 : .995;
        c.max_tries = seed%7 == 0 ? 2 : 20;
        c.rejection_probability_fallback = seed%7 == 0 ? 0 : .8;
        c.policy_temperature = seed%2 ? 1.6 : .7;
        int sizes[] = {5,7,11,15};
        int size=sizes[seed%4], canvas=size+seed%2;
        muzero::Game a(size,canvas,rule);
        reference::Board b(a);
        reference::BoardHistory hist;
        int player=1;
        Eval ea, eb;
        ea.mode=eb.mode=seed%7 == 0 ? 1 : seed%6;
        mt19937_64 ra(seed), rb(seed);
        auto result=muzero::initialize_opening(a,c,ea,ra);
        reference::Rand random(rb);
        reference::GomokuBalancedOpeningSettings settings;
        settings.avgDistFactor=c.avg_dist_factor;
        settings.balanceExponent=c.balance_exponent;
        settings.maxTries=c.max_tries;
        settings.rejectionProbability=c.rejection_probability;
        settings.rejectionProbabilityFallback=c.rejection_probability_fallback;
        reference::GomokuBalancedOpeningResult expected;
        if(c.probability>0 && random.nextBool(c.probability)) {
            reference::LegacyEvaluatorSelector select=[&](reference::GomokuBalancedOpening::PositionEvaluator& e) {
                random.nextBool(.5);
                e=[&](const reference::Board& bb,const reference::BoardHistory&, int p) { return eb.initial(bb.game.observation(p)).value; };
            };
            expected=reference::generateKataGomoImpl(b,hist,player,random,settings,select);
        }
        reference::NNEvaluator nn(eb);
        reference::Search search{&nn};
        bool run_policy = !expected.attempted || (expected.succeeded ? c.policy_after : c.policy_on_failure);
        if(c.policy_init && run_policy) reference::PlayUtils::initializeGameUsingPolicyKataGomo(&search,&search,b,hist,player,random,c.policy_init_mean,c.policy_temperature);
        if(result.actions!=hist.moves || a.observation()!=b.game.observation() || result.attempts!=expected.attempts ||
           result.start_value!=expected.startValue || result.failure!=expected.failureReason ||
           (result.status==muzero::OpeningStatus::Success)!=expected.succeeded ||
           ea.observations!=eb.observations || ra()!=rb()) {
            cerr << "Mismatch seed="<<seed<<" rule="<<int(rule)<<" calls="<<ea.calls<<","<<eb.calls<<" attempts="<<result.attempts<<","<<expected.attempts<<endl;
            return 1;
        }
        ++cases;
    }
    cout << "SkyZero source conformance: " << cases << " cases, identical actions, evaluations, values, retries, final boards and RNG states" << endl;
}
'''
with tempfile.TemporaryDirectory(prefix='muzero-opening-') as tmp:
    directory = Path(tmp)
    path = directory / 'compare.cpp'
    binary = directory / 'compare'
    path.write_text(text)
    subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-I' + str(root / 'cpp/include'),
                    '-I' + str(build / 'generated'), str(path), str(root / 'cpp/src/rules.cpp'),
                    str(root / 'cpp/src/search.cpp'), '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(build / 'test.cfg')], check=True, timeout=180)
