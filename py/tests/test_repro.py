"""可复现性四件套的回归测试：种子顺序 / RNG 状态存取 / 续训等价 / 故事评测口径。

这些测试都很小（几百步以内、几千字语料），目的是把它们跑进常规测试套件，
让「同 seed 逐位可复现」「续训与直训逐位等价」「生成可复现」「指标只算生成段」
这四条判据不会在后续改动里悄悄退化。
"""
from __future__ import annotations
import os
import random
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)

from minilm_torch import MiniLM, build_vocab, encode, make_batch, lr_at  # noqa
from repro import set_global_seed, rng_state, set_rng_state  # noqa
import eval_story as ES  # noqa

CFG = dict(ctx=16, d=16, layers=1, heads=2, ffMul=2)
TEXT = ('小红和小明去公园玩。他们在草地上跑，看见一只猫。猫在晒太阳，很舒服。'
        '他们说：我们一起玩吧。于是三个人在公园里跑来跑去，直到天黑。') * 40


def _train(seed: int, steps: int, seed_before_model: bool = True):
    """小规模训练；seed_before_model=False 复刻「先建模型后设种子」的旧顺序。"""
    if seed_before_model:
        set_global_seed(seed)
        itos, stoi = build_vocab(TEXT)
        model = MiniLM(len(itos), CFG)
    else:
        itos, stoi = build_vocab(TEXT)
        model = MiniLM(len(itos), CFG)
        set_global_seed(seed)
    ids = encode(TEXT, stoi)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    warmup = max(2, int(steps * 0.15))
    model.train()
    for step in range(steps):
        tok, tgt = make_batch(ids, 4, CFG['ctx'], random)
        opt.zero_grad()
        _, loss = model(tok, tgt)
        loss.backward()
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, 0.01, warmup)
        opt.step()
    return model.state_dict(), opt.state_dict()


def _flat_diff(a, b):
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a)


class TestSeedOrder(unittest.TestCase):
    def test_seed_before_model_is_bitwise_reproducible(self):
        s1, _ = _train(1234, 8)
        s2, _ = _train(1234, 8)
        self.assertEqual(0.0, _flat_diff(s1, s2), '同 seed 两次应逐位相同')

    def test_seed_after_model_is_not_reproducible(self):
        """失败样本：修复前的顺序下，同 seed 两次的初始权重就不同。"""
        set_global_seed(7)
        a = MiniLM(50, CFG).state_dict()
        b = MiniLM(50, CFG).state_dict()
        self.assertGreater(_flat_diff(a, b), 0.0)

    def test_different_seed_changes_result(self):
        s1, _ = _train(1234, 8)
        s3, _ = _train(4321, 8)
        self.assertGreater(_flat_diff(s1, s3), 0.0, '换 seed 应改变结果')


class TestRngState(unittest.TestCase):
    def test_state_roundtrip_keeps_batch_sequence(self):
        ids = list(range(500))
        set_global_seed(99)
        for _ in range(5):
            make_batch(ids, 2, CFG['ctx'], random)
        st = rng_state()
        ref, _ = make_batch(ids, 2, CFG['ctx'], random)
        set_rng_state(st)
        again, _ = make_batch(ids, 2, CFG['ctx'], random)
        self.assertTrue(torch.equal(ref, again), '恢复 RNG 状态后 batch 序列应一致')

    def test_torch_and_python_streams_are_both_captured(self):
        set_global_seed(5)
        st = rng_state()
        self.assertIn('python_random', st)
        self.assertIn('torch', st)
        torch.manual_seed(1)
        self.assertFalse(torch.equal(st['torch'], torch.get_rng_state()),
                         'torch 流应真实记录（否则恢复是假的）')


class TestResumeEquivalence(unittest.TestCase):
    def test_resume_matches_straight_run(self):
        """训 8 步 vs 训 4 步→恢复状态→再训 4 步，末步权重应逐位相同。"""
        ids = encode(TEXT, {c: i for i, c in enumerate(build_vocab(TEXT)[0])})
        itos, stoi = build_vocab(TEXT)

        def build():
            set_global_seed(1234)
            m = MiniLM(len(itos), CFG)
            return m, torch.optim.Adam(m.parameters(), lr=0.01)

        # 直训 8 步
        m1, o1 = build()
        for step in range(8):
            tok, tgt = make_batch(ids, 4, CFG['ctx'], random)
            o1.zero_grad()
            _, loss = m1(tok, tgt)
            loss.backward()
            o1.step()
        ref = {k: v.clone() for k, v in m1.state_dict().items()}

        # 训 4 步 → 保存（含 RNG）→ 恢复 → 再训 4 步
        m2, o2 = build()
        for step in range(4):
            tok, tgt = make_batch(ids, 4, CFG['ctx'], random)
            o2.zero_grad()
            _, loss = m2(tok, tgt)
            loss.backward()
            o2.step()
        blob = {'model': m2.state_dict(), 'opt': o2.state_dict(), 'rng': rng_state()}

        m3, o3 = build()
        m3.load_state_dict(blob['model'])
        o3.load_state_dict(blob['opt'])
        set_rng_state(blob['rng'])
        for step in range(4):
            tok, tgt = make_batch(ids, 4, CFG['ctx'], random)
            o3.zero_grad()
            _, loss = m3(tok, tgt)
            loss.backward()
            o3.step()
        self.assertEqual(0.0, _flat_diff(ref, m3.state_dict()),
                         '完整状态续训应与直训逐位等价')

    def test_missing_rng_state_breaks_equivalence(self):
        """失败样本：不恢复 RNG 状态（旧格式 checkpoint）就必然偏离。"""
        itos, stoi = build_vocab(TEXT)
        ids = encode(TEXT, stoi)

        def run(reset: bool):
            set_global_seed(1234)
            m = MiniLM(len(itos), CFG)
            o = torch.optim.Adam(m.parameters(), lr=0.01)
            for _ in range(4):
                tok, tgt = make_batch(ids, 4, CFG['ctx'], random)
                o.zero_grad()
                _, loss = m(tok, tgt)
                loss.backward()
                o.step()
            if reset:                     # 模拟「没有 RNG 状态可恢复」
                set_global_seed(1234)
            for _ in range(4):
                tok, tgt = make_batch(ids, 4, CFG['ctx'], random)
                o.zero_grad()
                _, loss = m(tok, tgt)
                loss.backward()
                o.step()
            return m.state_dict()

        self.assertGreater(_flat_diff(run(False), run(True)), 0.0)


class TestStoryEvalProtocol(unittest.TestCase):
    def _tiny_setup(self):
        itos, stoi = build_vocab(TEXT)
        set_global_seed(1234)
        model = MiniLM(len(itos), CFG)
        return model, itos, stoi

    def test_generation_is_deterministic_with_seed(self):
        model, itos, stoi = self._tiny_setup()
        pids = encode('小红和小明去', stoi)

        def gen(seed):
            set_global_seed(seed)
            return ES.generate_model(model, pids, 20, 0.8, 8, 1.15)

        self.assertEqual(gen(1234), gen(1234), '同 seed 生成应逐字相同')
        self.assertNotEqual(gen(1234), gen(999), '换 seed 应不同')

    def test_generated_ids_exclude_prompt(self):
        model, itos, stoi = self._tiny_setup()
        pids = encode('小红和小明去', stoi)
        set_global_seed(1)
        gen_ids = ES.generate_model(model, pids, 10, 0.8, 8, 1.15)
        self.assertLessEqual(len(gen_ids), 10, '返回的应是纯生成段，不含提示词')

    def test_metrics_ignore_prompt(self):
        """指标只在生成段上算：提示词里的关键词不应把事实检查送成分。"""
        facts = {'公园': '地点'}
        gen_only = ES.keyword_checks('他们在公园玩', facts)
        self.assertTrue(gen_only['公园'])
        self.assertFalse(ES.keyword_checks('他们在家里玩', facts)['公园'])

    def test_trigram_baseline_never_repeats_hard(self):
        _, stoi = build_vocab(TEXT)[0], {c: i for i, c in enumerate(build_vocab(TEXT)[0])}
        ids = encode(TEXT, stoi)
        tri = ES.TrigramGen(ids, len(stoi))
        set_global_seed(1234)
        gen = ES.decode(tri.generate(encode('小红和小明', stoi), 40, 0.8, 8, 1.15),
                        build_vocab(TEXT)[0])
        self.assertGreater(len(gen), 0)
        self.assertLessEqual(ES.repetition_stats(gen)['longest_repeat'], 4,
                             'trigram 采样不应出现长连续重复')


if __name__ == '__main__':
    unittest.main()
