# Guide to the Completed Results

[Project overview](../README.md) · [Reproducibility](REPRODUCIBILITY.en.md) · [Credits](CREDITS.en.md)

This guide summarizes the completed studies in English. Linked original reports retain their Chinese text, exact settings, individual responses, and review records. Results from different protocols should not be combined into a single leaderboard.

## Earlier training and negative findings

Claude Code assisted with pretraining and SFT, identity adaptation, GRPO, math tool-use RL, DPO, MoE, white-box distillation, black-box distillation, and RL after distillation. The project uses upstream MiniMind architecture and training code, with project-specific orchestration and evaluation.

Several observations are important even though they did not produce a better general assistant:

- The longer GRPO run increased total reward, but most of the increase came from formatting rather than a clearly established content improvement.
- The higher-learning-rate DPO variant performed better on the preference metric while producing worse responses under generation checks.
- The tested MoE and MoE-teacher white-box-distillation recipes did not deliver broad gains across loss, reward, and generation speed.
- Merging an identity LoRA into a different base severely damaged math performance under the historical scoring protocol. Adapters must be trained and merged against the same actual base.

The [historical report](../my_eval/README.md) preserves these runs. Its early 200-question math scores use a legacy number-inclusion metric and must not be presented as current strict correctness.

## Official-model comparison

On 296 held-out answers with valid answer-token coverage, the official model had lower prediction loss than the local SFT baseline: approximately 2.0927 versus 2.4583. The paired improvement was about 0.366 ± 0.021 loss units. The reward-model chat comparison did not show a clear advantage.

Lower prediction loss therefore supports a narrower conclusion than “better general chat.” Whether the official model encountered the public evaluation material during upstream training is unknown. See the [official comparison](../my_eval/results/official_compare_20261024/REPORT.md).

## Identity adaptation and v2

The accepted identity-repair candidate is `bb_rl_id_anchor4`, exported as `MiniMind-YuyangLU-v2`. In its identity regression set, the candidate matched the customizer's name in 20 of 21 responses, compared with 18 of 21 for the previous identity version. Its held-out answer loss was 2.4964, compared with 2.6043 for that previous version and 2.4888 for the unpatched RL base.

Name matching is only a proxy. Some responses that contained the expected names still failed to explain the upstream-author/local-customizer relationship. The report separates string matches from full-content review and retains ordinary-response identity leakage. See the [identity study](../my_eval/results/identity_repair_20261024/REPORT.md).

## Continued-distillation controls and corrected grading

Two controls start from the same `bb_dist` weights. One continues learning from old teacher questions; the other substitutes a new batch of teacher trajectories. Their training-row counts and update counts are matched. Answer lengths differ, so this is not an equal-supervised-token or equal-compute comparison.

The follow-up audit found three known exact math-question overlaps and one chat-question overlap. Main comparisons exclude those examples consistently while retaining the original 200-question and 30-question records.

Strict full-answer scoring on 197 questions × 2 samples gives:

| Model | Meaning | Strict pass rate |
|---|---|---:|
| `bb_dist` | Original distilled model | 57.61% |
| `bb_rl` | Distillation followed by RL | 68.27% |
| `bb_sft_replay` | Continued distillation using old questions | 70.05% |
| `bb_sft_new` | Continued distillation using new questions | 72.84% |

The latter three improve over the original distilled model under this protocol, but their pairwise 95% intervals include zero. Their exact ordering is unresolved; equivalence is not established. Extra updates and changed data composition also prevent attributing the original improvement entirely to RL.

The two continued-distillation controls have worse held-out answer loss, around 2.541 versus 2.489 for `bb_rl`. Math performance alone does not establish a better overall assistant.

The corrected grader requires complete answers with consistent numerical claims and expression associations. Merely finding a target number inside a longer number, a repeated question, or a contradictory answer is insufficient. Unknown formats are reviewed rather than silently treated as incorrect. This was a retrospective correction informed by examining historical outputs, not a newly preregistered blind test. See the [control study](../my_eval/results/distill_control_20261025/REPORT.md) and [strict-math correction](../my_eval/results/strict_math_20261026/REPORT.md).

## Reward-based candidate selection

For v2, one fixed pool of four candidates per question supports first-candidate, best-of-2, and best-of-4 comparisons without changing generation luck between methods. The reward selector receives the conversation and final answer, not the reference answer or calculator result.

| Selection rule | Strict math pass rate, 197 questions |
|---|---:|
| First candidate | 53.81% |
| Reward-selected from two | 65.99% |
| Reward-selected from four | 70.56% |
| First complete, scoreable candidate | 54.31% |

Best-of-4 gains 16.75 ± 3.36 percentage points over the first candidate; its 95% interval is [10.15, 23.35]. The extra gain over best-of-2 remains uncertain: its interval includes zero. Four-candidate generation plus selection took roughly five times the measured generation/scoring time on math in this setup.

The feature is available in the local demo and defaults to off. This study evaluates one model and its fixed candidate pool; its results do not automatically apply to the later models, other tool configurations, or long conversations. See the [best-of-N study](../my_eval/results/best_of_n_20261026/REPORT.md).

## Tool selection: failed first attempt, then v3

The [first tool-selection study and confirmation tests](../my_eval/results/tool_choice_20261026/REPORT.md) did not establish a robust improvement. Those negative results and responses remain in the archive.

The second study varies tool count, tool descriptions, order, and calculation wording, and includes numerical text that should be copied without invoking a calculator. It improves first-turn correctness of both calculator selection and every requested expression argument from 42.53% to 79.51%. Nonempty ordinary responses without tool calls improve from 26.74% to 99.65%.

There are 48 base math questions and 24 base ordinary questions, each expanded across 12 tool conditions. Statistical comparisons aggregate within each base question. The 864 expanded responses per model are not 864 independent questions.

The accepted model is `tool_choice2b_e3`, exported as v3. The evidence concerns calculator selection versus no call with fixed tool names and schemas; other tools are distractors. It does not establish arbitrary tool use or real-time weather/exchange-rate services. See the [second study](../my_eval/results/tool_choice_round2_20261026/REPORT.md).

## Chat repair of MiniMind itself: v4

The accepted model is `chat_repair_id_e3`, exported as v4. Starting from v3, full-parameter training uses curated teacher demonstrations and fixed examples, together with math supervision and frozen-model retention. An identity refinement is trained and merged only against its exact base, with parameter reconstruction checks. The teacher supplies some training demonstrations and is not used as the runtime chat backend.

Formal evaluation uses 48 base questions in four categories, each under three conditions: no tools, six original-description tools, and six tools with previously known description rewrites. Each model produces 144 responses. Thinking and best-of-N are disabled.

| Metric | v3 | v4 |
|---|---:|---:|
| Mean content score, 0–2 | 0.389 | 0.924 |
| Fully usable responses, score 2 | 11.11% | 37.50% |
| Nonempty, complete responses without calls | 83.33% | 98.61% |
| Historical strict math, 197 questions | 71.57% | 82.74% |
| Held-out answer loss, lower is better | 2.49630 | 2.52263 |

Content-score gain is 0.535 ± 0.096 points, with a 95% interval of [0.354, 0.722]. Usability gain is 26.39 ± 5.61 percentage points, with a 95% interval of [15.97, 37.50]. Improvement also remains positive after excluding identity questions, so identity-name repetition alone does not account for acceptance.

The loss increase is real: +0.02633 ± 0.00666, with a 95% interval of approximately [0.01221, 0.03823]. It stays within the predeclared allowance of 0.05; it must not be described as “unchanged language ability.” Fully usable identity responses rise from 12 to 13 out of 14, with one relationship explanation still incomplete.

The candidate passes this experiment's predefined acceptance gates and its export and real Streamlit UI checks. Nevertheless, 62.50% of responses are not fully usable. Limited topic structures, one training seed, fixed generation seeds, and assistant content review restrict generalization. This is not evidence of a generally reliable assistant. See the [chat-repair study](../my_eval/results/chat_repair_20261027/REPORT.md).

## Distillation from a larger teacher: v5

This study asks two separate questions: does training on screened teacher answers improve v4, and does a 7B teacher teach better than a 1.5B teacher? The first passes its predeclared acceptance gates. The second is not established.

Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct answered the same 1,000 questions with one fixed prompt and deterministic generation. Retrieval and format questions required the exact expected output. Open-ended answers received individual review with teacher labels hidden.

| Accepted answers | 1.5B | 7B |
|---|---:|---:|
| Retrieval, exact output | 205/400 | 351/400 |
| Format instructions, exact output | 70/100 | 77/100 |
| Everyday explanations | 149/200 | 192/200 |
| Short-message writing | 181/300 | 254/300 |
| Total | 605/1000 | 874/1000 |

Among rejected exact-output answers, 151 from the 1.5B teacher and 48 from the 7B teacher differ only in punctuation, case, or spacing. The strict rule was set in advance and was not relaxed.

Both training sets use only the 561 questions accepted from both teachers (196 retrieval, 161 writing, 143 explanation, 61 format), with the same prompts, tool menus, and retention rows. Teacher answers are not rewritten. Each arm has 2,602 rows and trains all parameters from v4 for 1,302 updates, anchored to frozen v4 weights. The 7B answers total 6,201 tokens versus 8,066, so the design matches questions and updates, not answer volume. Development data selected epoch 2 in both arms before weights and evaluation files were locked.

Formal evaluation uses 96 new questions, 24 per category, under three tool conditions, for 288 responses per model. Thinking and best-of-N are disabled.

| Metric | v4 | 1.5B arm | 7B arm, v5 |
|---|---:|---:|---:|
| Mean content score, 0–2 | 0.378 | 0.622 | 0.681 |
| Fully usable responses, score 2 | 11.46% | 26.04% | 28.47% |
| Nonempty, complete responses without calls | 100% | 99.31% | 100% |
| Historical strict math, 197 questions | 82.74% | 86.29% | 86.29% |
| Reworded-tool strict math, 48 questions | 77.08% | 75.00% | 79.17% |
| Held-out answer loss, lower is better | 2.52263 | 2.52168 | 2.52476 |
| Fully usable identity responses | 13/14 | 14/14 | 12/14 |

| Comparison | Content score | Usable percentage points |
|---|---:|---:|
| 7B arm − v4 | +0.302 ± 0.079, [+0.153, +0.455] | +17.01 ± 4.19, [+9.03, +25.35] |
| 1.5B arm − v4 | +0.243 ± 0.076, [+0.101, +0.396] | +14.58 ± 3.97, [+6.94, +22.57] |
| 7B arm − 1.5B arm | +0.059 ± 0.039, [−0.021, +0.135] | +2.43 ± 1.93, [−1.39, +6.25] |

| Usable rate by category | v4 | 1.5B arm | 7B arm, v5 |
|---|---:|---:|---:|
| Format instructions | 20.8% | 66.7% | 65.3% |
| Retrieval | 16.7% | 26.4% | 27.8% |
| Short-message writing | 6.9% | 9.7% | 16.7% |
| Everyday explanations | 1.4% | 1.4% | 4.2% |

The candidate from the 7B arm passes all nine acceptance gates and its export and real Streamlit UI checks, so it is released as v5. The evidence still limits what that means:

- **The teacher comparison is unresolved.** Averages favor the 7B arm, but both intervals include zero, and one training seed was run. This does not establish that the teachers are equivalent either.
- **The matched design is a narrow test.** It compares different teachers' answers to the same questions, so the 7B teacher's extra accepted answers go unused. Training on all accepted answers would need a separate experiment.
- **Most of the gain is in format instructions.** Their templates also appear in training, although the question text does not overlap. Without them, the 7B arm rises from 8.33% to 16.20% usable, a gain of 7.87 ± 3.98 percentage points with a 95% interval of [0.46, 15.74].
- **Retention is not improvement.** Against v4, historical math changes by +3.55 ± 1.95 percentage points, [0.00, 7.61]; reworded-tool math by +2.08 ± 4.70, [−6.25, 10.42]; and loss by +0.00213 ± 0.00411, [−0.00664, 0.00913]. Identity falls by one response, the most the gates allow.
- **Results are not comparable across question sets.** v4 scores 37.50% in its own study and 11.46% here because the questions differ.
- **v5 remains weak.** 71.53% of its responses are not fully usable, and everyday explanations reach only 4.2%. Content received assistant review with model and teacher labels hidden, not independent human blind evaluation.

The cloud run also found two practical problems. Downloads through a Hugging Face proxy and a mirror were too slow, so the weights came from ModelScope and their SHA-256 hashes were compared with the pinned Hugging Face revision. The automatic-shutdown call failed with `Exec format error` after results were saved, so the instance was shut down manually. Mock tests had not executed that call, and the script must be fixed before another cloud run. See the [larger-teacher report](../my_eval/results/strong_teacher_20261027/REPORT.md).

## Common interpretation rules

Results are evidence about the particular model, data, and recipe tested. Prediction loss, reward, tool routing, full-answer math consistency, name matching, and content usability measure different things. Exact overlap removal does not rule out semantic or upstream training exposure. Assistant review is documented but is not independent human blind evaluation. Retained failures and uncertainty are part of the findings, not exceptions to omit.
