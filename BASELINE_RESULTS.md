# Baseline Results: Qwen3-1.7B on LeetCode

**Model:** `Qwen/Qwen3-1.7B` (non-thinking mode, greedy decoding)  
**Dataset:** LeetCode test split (228 problems)  
**Date:** 2026-02-26  

## Overall

| Metric | Value |
|---|---|
| **pass@1** | **13.60%** (31/228) |
| **mean test pass rate** | **27.54%** |
| Wall time | 38.1 min (MPS, Apple Silicon) |

## By Difficulty

| Difficulty | pass@1 | Mean Test Pass Rate | n |
|---|---|---|---|
| Introductory (Easy) | **39.58%** | 60.42% | 48 |
| Interview (Medium) | **8.91%** | 25.15% | 101 |
| Competition (Hard) | **3.80%** | 10.63% | 79 |

## Settings

| Setting | Value |
|---|---|
| Decoding | Greedy (`do_sample=False`) |
| dtype | float16 |
| Max new tokens | 512 |
| Exec timeout | 5s per test case |
| Max test cases | 10 per problem |
| Thinking mode | Disabled (`enable_thinking=False`) |

## Notes

- This is the **pre-training baseline** — no SDFT, SDPO, or any fine-tuning applied.
- Clean difficulty gradient (Easy >> Medium >> Hard) confirms room for improvement via distillation.
- All LeetCode problems are function-call style (`Solution().method()`), converted to APPS format for unified evaluation.
- Full per-problem results saved in `logs/baseline_qwen3_leetcode.json`.
