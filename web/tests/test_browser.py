import os
import unittest


@unittest.skipUnless(os.environ.get('MUZERO_WEB_TEST_URL'), 'Set MUZERO_WEB_TEST_URL')
class BrowserTests(unittest.TestCase):
    def test_real_game_in_browser(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1360, 'height': 1100}, device_scale_factor=1)
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(os.environ['MUZERO_WEB_TEST_URL'])
            page.wait_for_function("document.querySelector('#model').options.length > 0")
            page.locator('#size').select_option('7')
            page.locator('#rule').select_option('standard')
            page.locator('#visits').fill('32')
            page.locator('#new-game').click()
            page.wait_for_function("document.querySelector('#status').textContent === '轮到你落子'")
            page.locator('[data-action="24"]').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 2 手'")
            page.wait_for_function("!document.querySelector('#undo').disabled")
            self.assertTrue(page.locator('#heatmap-data').is_visible())
            self.assertEqual(page.locator('#prior-map .heat-cell').count(), 49)
            self.assertEqual(page.locator('#prior-map .heat-stone').count(), 1)
            self.assertEqual(page.locator('#board .stone').count(), 2)
            snapshot = page.evaluate("async () => (await (await fetch('/api/state')).json()).analysis")
            for field, selector in [('network_prior', '#prior-map'), ('visit_policy', '#search-map')]:
                values = page.locator(selector + ' .heat-cell').evaluate_all("cells => cells.map(cell => Number(cell.dataset.value))")
                self.assertAlmostEqual(sum(values), 1)
                for candidate in snapshot['candidates']:
                    self.assertAlmostEqual(values[candidate['action']], candidate[field])
            page.locator('#search-map-kind').select_option('selection_weight')
            selected = page.locator('#search-map .heat-cell').evaluate_all("cells => cells.map(cell => Number(cell.dataset.value))")
            for candidate in snapshot['candidates']:
                self.assertAlmostEqual(selected[candidate['action']], candidate['selection_weight'])
            page.locator('#prior-map .heat-cell').first.hover()
            self.assertIn('网络先验', page.locator('#heatmap-detail').inner_text())
            page.locator('#heatmaps').screenshot(path='/tmp/muzero-heatmaps.png')
            page.locator('#undo').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 0 手'")
            self.assertTrue(page.locator('#heatmap-data').is_hidden())
            self.assertEqual(page.locator('#prior-map .heat-cell').count(), 0)
            for _ in range(4):
                page.wait_for_function("document.querySelector('#status').textContent === '轮到你落子'")
                turn = int(page.locator('#move-count').inner_text().split()[1])
                page.locator('.board-point:not([disabled])').first.click()
                page.wait_for_function("n => Number(document.querySelector('#move-count').textContent.split(' ')[1]) >= n", arg=turn + 2)
            page.wait_for_function("!document.querySelector('#undo').disabled")
            page.screenshot(path='/tmp/muzero-web-desktop.png', full_page=True)
            count = page.locator('#move-count').inner_text()
            page.reload()
            page.wait_for_function("text => document.querySelector('#move-count').textContent === text", arg=count)
            page.locator('#size').select_option('5')
            page.locator('#rule').select_option('renju')
            page.locator('input[name="human"][value="-1"]').check()
            page.locator('#new-game').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 1 手'")
            page.wait_for_function("document.querySelector('#status').textContent === '轮到你落子'")
            self.assertTrue(page.locator('#undo').is_disabled())
            page.locator('.board-point:not([disabled])').first.click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 3 手'")
            page.wait_for_function("!document.querySelector('#undo').disabled")
            page.locator('#undo').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 1 手'")
            for _ in range(25):
                page.wait_for_function("!document.querySelector('#new-game').disabled")
                status = page.locator('#status').inner_text()
                if '获胜' in status or '和棋' in status:
                    break
                page.locator('.board-point:not([disabled])').first.click()
                page.wait_for_function("!document.querySelector('#new-game').disabled")
            else:
                self.fail('Game did not finish')
            self.assertEqual(page.locator('.board-point:not([disabled])').count(), 0)
            page.set_viewport_size({'width': 390, 'height': 844})
            page.screenshot(path='/tmp/muzero-web-mobile.png', full_page=True)
            self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
            self.assertEqual(errors, [])
            browser.close()


if __name__ == '__main__':
    unittest.main()
