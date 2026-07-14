Read ImplementationPLAN.txt and IMPLEMENTATION_STATUS.md.

This session finalizes the production classifier semantics, calibration policy,
and inference flow before backend integration.

The current three-tier opinion classification should no longer be treated as a
production requirement. Implement a simpler architecture with:

1. Binary bias classification with three calibrated decision states:
   - no_clear_bias
   - possible_bias
   - clear_bias

2. Binary writing-style classification with three calibrated decision states:
   - objective_style
   - uncertain
   - opinionated_style

3. Evidence-based factuality assessment only after claim extraction and
   retrieval:
   - supported
   - contradicted
   - insufficient_evidence
   - not_verifiable
   - not_assessed

Do not treat writing style as factual correctness.

Before editing, inspect the current training, calibration, inference, retrieval,
backend, schema, and test implementations. Then show me the exact files and
functions you intend to modify. Wait for approval before applying changes.

======================================================================
A. REQUIRED MODEL ARCHITECTURE
======================================================================

Encode each sentence or sentence-context window once with the shared encoder:

    embedding = encoder(sentence_with_context)

Feed the same embedding independently into two production heads:

    bias_logit = bias_head(embedding)
    opinion_style_logit = opinion_style_head(embedding)

The two heads must operate in parallel.

The opinion prediction must not be passed into the bias head, and the bias
prediction must not be passed into the opinion head.

Production probabilities:

    p_bias = sigmoid(bias_logit)
    p_opinionated_style = sigmoid(opinion_style_logit)

The existing first hierarchical opinion stage currently named something like:

    q_nonfactual

must be renamed semantically to:

    q_opinionated_style

or:

    p_opinionated_style

It predicts writing style, not factual truth.

For backward compatibility, old artifact fields or checkpoints may recognize
q_nonfactual as an alias, but all new manifests, APIs, logs, and user-visible
outputs must use accurate opinion-style terminology.

The existing second hierarchical stage:

    q_writer_given_nonfactual

distinguishing:

    Somewhat factual but also opinionated
    versus
    Expresses writer's opinion

must not be used in the production pipeline.

It may remain available only behind an explicit experimental or legacy option.
It must not influence:

- bias classification;
- report-trigger selection;
- retrieval;
- factuality assessment;
- production API opinion-style output;
- artifact promotion gates.

======================================================================
B. TRAINING TARGETS
======================================================================

Bias head target:

    label == 0 -> not biased
    label == 1 -> biased

Opinion-style head target:

    Entirely factual
        -> objective_style, target 0

    Somewhat factual but also opinionated
        -> opinionated_style, target 1

    Expresses writer's opinion
        -> opinionated_style, target 1

    NaN, No agreement, invalid, or unrecognized labels
        -> masked from opinion-style loss

"Entirely factual" means objective or factual writing style. It does not mean
that the claim has been externally verified as true.

Normalize opinion strings before target mapping:

- Unicode NFKC normalization;
- curly and straight apostrophe normalization;
- trimming;
- whitespace collapsing;
- case-insensitive matching.

Keep the bias and opinion-style losses independent, aside from sharing the
encoder.

Report separate training and validation metrics for:

- binary bias;
- binary opinion style.

Do not report the deprecated middle-versus-writer distinction as the primary
production opinion metric.

======================================================================
C. PROBABILITY CALIBRATION
======================================================================

Use only the dedicated calibration partition.

Never fit calibrators or select thresholds using:

- training data;
- development/validation data;
- locked test data.

Fit and store separate probability calibrations for:

1. p_bias
2. p_opinionated_style

Use the repository's existing reliable binary calibration method. Prefer
temperature or logistic/Platt calibration over adding a complex new calibration
method. Do not silently apply sigmoid or calibration twice.

Persist the calibration method, parameters, split-manifest hash, model hash,
head type, label mapping, sample support, and schema version in the calibration
artifact.

Validate this binding during inference. Fail explicitly on incompatible model,
label-map, split, or calibration artifacts unless an explicit legacy mode was
requested.

======================================================================
D. BIAS DECISION THRESHOLDS
======================================================================

Bias states represent model confidence, not bias severity.

Do not describe possible_bias as objectively "mild bias."

Store two independently selected thresholds:

    bias_no_clear_max
    bias_clear_min

Decision rule:

    p_bias <= bias_no_clear_max
        -> no_clear_bias

    p_bias >= bias_clear_min
        -> clear_bias

    otherwise
        -> possible_bias

Select bias_clear_min as the lowest calibrated probability threshold satisfying:

- one-sided 95% Wilson lower confidence bound for positive predictive value
  / biased-class precision >= configured target, default 0.90;
- configured minimum predicted support, default at least 30 examples.

Choosing the lowest qualifying threshold maximizes clear-bias coverage while
maintaining the precision requirement.

Select bias_no_clear_max as the highest calibrated probability threshold
satisfying:

- one-sided 95% Wilson lower confidence bound for negative predictive value
  / not-biased prediction precision >= configured target, default 0.90;
- configured minimum predicted support, default at least 30 examples.

Choosing the highest qualifying threshold maximizes confident no-bias coverage.

Require:

    bias_no_clear_max < bias_clear_min

If the calibration data cannot support one of these confident regions:

- do not invent a threshold;
- do not lower the configured confidence target silently;
- disable that confident state;
- route affected examples to possible_bias;
- record the calibration failure clearly.

Persist threshold support, precision, Wilson lower bound, and coverage.

======================================================================
E. OPINION-STYLE DECISION THRESHOLDS
======================================================================

Store two independent thresholds:

    opinion_objective_max
    opinion_opinionated_min

Decision rule:

    p_opinionated_style <= opinion_objective_max
        -> objective_style

    p_opinionated_style >= opinion_opinionated_min
        -> opinionated_style

    otherwise
        -> uncertain

Select opinion_opinionated_min as the lowest threshold satisfying:

- one-sided 95% Wilson lower confidence bound for opinionated-style precision
  >= configured target;
- configured minimum support.

Select opinion_objective_max as the highest threshold satisfying:

- one-sided 95% Wilson lower confidence bound for objective-style prediction
  precision / negative predictive value >= configured target;
- configured minimum support.

Use configurable targets. Default to 0.85 or 0.90, depending on the existing
calibration policy, but record the exact value in the artifact.

Require:

    opinion_objective_max < opinion_opinionated_min

If either confident region lacks sufficient calibration support, return
uncertain rather than fabricating a confident style label.

Do not use the old middle-versus-writer head or threshold to produce these
states.

======================================================================
F. PRODUCTION INFERENCE FLOW
======================================================================

Implement or verify this sequence:

1. Segment the article into target sentences and surrounding-context windows.

2. Encode each selected window once.

3. Run the independent bias and opinion-style heads in parallel.

4. Apply separately calibrated probabilities and thresholds.

5. Produce:

       bias_assessment:
           no_clear_bias
           possible_bias
           clear_bias

       opinion_style:
           objective_style
           uncertain
           opinionated_style

6. Use bias assessment, not opinion style, for report candidate selection.

Default report-trigger behavior:

- clear_bias:
    eligible for report generation.

- possible_bias:
    retained as an uncertain candidate;
    excluded from strong claims by default;
    optionally included when explicitly configured.

- no_clear_bias:
    not selected for a bias report by default.

7. Opinion style may be displayed as auxiliary context, but it must not gate:

- evidence retrieval;
- factuality assessment;
- report eligibility;
- bias status.

8. For selected bias candidates, determine whether the sentence contains a
verifiable factual claim.

9. Only verifiable claims proceed to hybrid web/Qdrant retrieval.

10. Pass the selected sentence, surrounding context, calibrated bias state,
opinion style, extracted claim, and retrieved evidence into report generation.

11. Produce evidence status only after evaluating retrieved material:

       supported
       contradicted
       insufficient_evidence
       not_verifiable
       not_assessed

Semantic similarity alone must not be treated as support.

12. If retrieval fails or evidence is inadequate, return
insufficient_evidence or not_assessed. Never fabricate citations.

======================================================================
G. PUBLIC API SEMANTICS
======================================================================

Expose separate fields similar to:

    bias_assessment:
        "no_clear_bias" |
        "possible_bias" |
        "clear_bias"

    opinion_style:
        "objective_style" |
        "uncertain" |
        "opinionated_style"

    evidence_status:
        "supported" |
        "contradicted" |
        "insufficient_evidence" |
        "not_verifiable" |
        "not_assessed"

Internal calibrated scores may remain available in diagnostic metadata, but the
UI should not present a probability as a literal amount or severity of bias.

Do not expose a classifier field named:

- factuality;
- factual;
- nonfactual;
- verified true;
- verified false;

unless that field is explicitly generated by the evidence-assessment stage.

Preserve existing API fields when practical, but deprecate semantically
incorrect fields explicitly rather than silently changing their meaning.

======================================================================
H. CHECKPOINT SELECTION AND EVALUATION
======================================================================

Production checkpoint selection should use clean development-set metrics for:

- bias macro-F1;
- biased-class precision and recall;
- opinion-style binary macro-F1;
- objective-style recall;
- opinionated-style recall;
- validation loss.

Do not use the deprecated reconstructed three-class opinion probabilities as the
primary checkpoint-selection metric.

Do not tune decision thresholds during ordinary validation epochs.

Threshold selection happens after checkpoint selection and uses only the
dedicated calibration split.

Report compact terminal summaries. Save full diagnostics as formatted JSON.

Required evaluation outputs:

Bias:
- accuracy;
- balanced accuracy;
- macro-F1;
- per-class precision, recall, and F1;
- confusion matrix;
- ROC-AUC;
- PR-AUC;
- threshold support and coverage;
- Wilson lower bounds.

Opinion style:
- the same binary metrics;
- objective/opinionated/uncertain coverage after calibration.

Final decision-state coverage:
- no_clear_bias count and percentage;
- possible_bias count and percentage;
- clear_bias count and percentage;
- objective_style count and percentage;
- uncertain opinion count and percentage;
- opinionated_style count and percentage.

The locked test split must be evaluated only after model and thresholds are
fixed.

======================================================================
I. REQUIRED TESTS
======================================================================

Add or update tests verifying:

1. A single encoder embedding feeds both heads independently.

2. Valid BABE opinion labels map correctly:
   - Entirely factual -> objective
   - both other agreed tiers -> opinionated.

3. NaN and No agreement are masked only from the opinion-style loss.

4. Curly and straight apostrophes normalize identically.

5. Sigmoid and probability calibration are each applied exactly once.

6. Bias upper threshold selection enforces positive-precision Wilson LCB.

7. Bias lower threshold selection enforces negative-prediction precision
   / NPV Wilson LCB.

8. Opinion upper and lower thresholds enforce their respective precision
   requirements.

9. Threshold search never reads validation or test labels.

10. Threshold behavior is deterministic.

11. If support is insufficient, the result becomes possible_bias or uncertain
    rather than a fabricated confident classification.

12. Opinion style cannot change bias classification.

13. Opinion style cannot trigger or suppress retrieval.

14. The deprecated middle-versus-writer head is absent from normal production
    output.

15. No classifier-only path emits supported or contradicted.

16. Evidence status is generated only after retrieval/evidence assessment.

17. Existing artifacts either load through an explicit compatibility alias or
    fail with a clear compatibility error.

======================================================================
J. NON-GOALS
======================================================================

Do not:

- retrain the model in this session;
- tune thresholds on the test set;
- restore three-way opinion classification as a production dependency;
- call possible_bias "mild bias";
- equate objective writing style with truth;
- allow opinion style to gate retrieval;
- redesign the completed hybrid retrieval implementation;
- migrate the LLM provider;
- add unrelated UI features;
- perform broad repository refactoring.

======================================================================
K. DELIVERABLES
======================================================================

Before implementation, provide:

1. the current pipeline traced from code;
2. every semantic mismatch found;
3. exact files and functions to modify;
4. proposed calibration-artifact schema;
5. proposed API field changes;
6. backward-compatibility plan.

After approval and implementation:

1. run focused unit tests;
2. run relevant regression tests;
3. run py_compile checks;
4. do not launch full training;
5. update IMPLEMENTATION_STATUS.md;
6. provide the exact next training and calibration commands;
7. list any remaining code paths that still misuse "factual" to imply verified
   truth.