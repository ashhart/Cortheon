"""Executed adapter checks for evaluator intervention profiles."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from cortheon.qualification_core.conditions import execution_profile


def _profile(condition: str) -> dict:
    profile = execution_profile(condition, "a" * 64)
    profile["nonce"] = "3" * 32
    return profile


def _node(
    script: str,
    profile: dict,
    *,
    strip_types: bool = False,
    cwd: Path | None = None,
) -> dict:
    command = ["node"]
    if strip_types:
        command.append("--experimental-strip-types")
    command.extend(["--input-type=module", "-e", script])
    completed = subprocess.run(
        command,
        env={
            **os.environ,
            "CORTHEON_EVALUATOR_PROFILE": json.dumps(profile),
            "CORTHEON_COGNITIVE_TOKEN": "test-secret-token",
            "CORTHEON_EVALUATOR_MAX_STEPS": "4",
            "CORTHEON_AUTO_ENABLE": "1",
            "CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1",
            "CORTHEON_MAX_HOST_TOOL_CALLS": "12",
        },
        text=True,
        capture_output=True,
        check=False,
        cwd=cwd,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _stage_pi(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "src" / "cortheon" / "pi_core"
    shutil.copytree(source, tmp_path / "pi_core")
    scope = tmp_path / "node_modules" / "@earendil-works"
    scope.mkdir(parents=True)
    coding = scope / "pi-coding-agent"
    coding.mkdir()
    (coding / "package.json").write_text(
        json.dumps({"type": "module", "exports": "./index.js"}),
        encoding="utf-8",
    )
    (coding / "index.js").write_text(
        "export const createFindTool=()=>({});"
        "export const createGrepTool=()=>({});"
        "export const createReadTool=()=>({});\n",
        encoding="utf-8",
    )
    pi_ai = scope / "pi-ai"
    pi_ai.mkdir()
    (pi_ai / "package.json").write_text(
        json.dumps(
            {
                "type": "module",
                "exports": {".": "./index.js", "./compat": "./compat.js"},
            }
        ),
        encoding="utf-8",
    )
    (pi_ai / "index.js").write_text(
        "export const uuidv7=()=> '00000000-0000-7000-8000-000000000000';\n",
        encoding="utf-8",
    )
    (pi_ai / "compat.js").write_text(
        "export const complete=async()=>({content:[]});\n",
        encoding="utf-8",
    )
    return tmp_path


def test_pi_verification_only_preserves_prompt_and_does_not_bind_tools(tmp_path) -> None:
    script = r"""
      import {registerSessionEvents} from './pi_core/session_events.ts';
      import {registerToolEvents} from './pi_core/tool_events.ts';
      import {
        scheduleContinuation, setActive, setEnabled,
      } from './pi_core/state.ts';
      const handlers = {};
      const pi = {
        on: (name, callback) => { handlers[name] = callback; },
        sendUserMessage: () => {}, getActiveTools: () => ['edit'],
        sendMessage: () => {},
      };
      registerSessionEvents(pi);
      registerToolEvents(pi);
      setEnabled(true);
      setActive({
        sessionId: 'vx_test', goal: 'Fix app.py and run pytest -q',
        deliverable: 'code_change', completed: false, evidenceIds: ['ev1'],
        mutationTargets: ['app.py'], protectedTestPaths: ['test_app.py'],
        repairPlan: {path: 'app.py', oldText: 'bad', newText: 'good'},
        testInvocation: {commandLine: 'pytest -q'}, admittedToolCalls: 0,
        redundantDiscoveryCalls: 0, automaticContinuations: 0,
      });
      scheduleContinuation('continue');
      const prompt = await handlers.before_agent_start(
        {prompt: 'continue', systemPrompt: 'base'},
        {isProjectTrusted: () => true, abort: () => {}},
      );
      const input = {path: 'other.py'};
      const decision = await handlers.tool_call(
        {toolName: 'edit', input},
        {isProjectTrusted: () => true, abort: () => {}},
      );
      console.log(JSON.stringify({
        prompt: prompt.systemPrompt, decision: decision ?? null, input,
      }));
    """
    result = _node(
        script,
        _profile("verification_only"),
        strip_types=True,
        cwd=_stage_pi(tmp_path),
    )
    assert result["prompt"] == "base"
    assert result["decision"] is None
    assert result["input"] == {"path": "other.py"}


def test_opencode_verification_only_never_runs_repairs_tests_or_research() -> None:
    script = r"""
      import {createSystemTransformHook} from './src/cortheon/opencode_core/hook_conversation.js';
      import {createToolAfterHook, createTextCompleteHook} from './src/cortheon/opencode_core/hook_output.js';
      import {investigations} from './src/cortheon/opencode_core/state.js';
      const calls = {repair: 0, test: 0, research: 0, prompt: 0};
      const state = {
        automatic: true, active: true, cortheonSessionID: 'vx_test',
        goal: 'Fix app.py and run pytest -q', deliverable: 'code_change',
        requestedTestCommand: 'pytest -q', mutated: true, evidenceIDs: [],
      };
      investigations.set('s', state);
      const noop = async (_id, value) => value;
      const transform = createSystemTransformHook({
        runtimeBase: 'http://runtime', ensureAutomaticInvestigation: async () => state,
        acquireRequestedEvidence: async () => false,
        submitAutomaticObservation: noop, ensureCausalChain: noop,
        resolveCounterexampleRequest: noop,
        attemptBoundedMultiRepair: async (_id, value) => { calls.repair++; return value; },
        attemptBoundedAutomaticRepair: async (_id, value) => { calls.repair++; return value; },
        certifyDeterministicResearch: noop, resyncEvidenceFromRuntime: noop,
        ensureSemanticEvidence: noop, submitAutomaticCompletion: noop,
        runtimeCall: async () => ({}),
      });
      const prompt = {system: ['base']};
      await transform['experimental.chat.system.transform']({sessionID: 's'}, prompt);
      const after = createToolAfterHook({
        debug: async () => {}, captureMutationAfter: async () => {},
        runRequestedTest: async () => { calls.test++; return {}; },
        patchHygieneIssue: async () => undefined, certifyCodeChange: noop,
        submitAutomaticObservation: noop, submitPassiveObservations: noop,
      });
      await after['tool.execute.after'](
        {sessionID: 's', tool: 'edit', args: {filePath: 'app.py'}},
        {output: 'updated'},
      );
      const complete = createTextCompleteHook({
        acquireRequestedEvidence: async () => false,
        submitAutomaticObservation: noop, resyncEvidenceFromRuntime: noop,
        ensureCausalChain: noop, resolveCounterexampleRequest: noop,
        ensureSemanticEvidence: noop,
        attemptBoundedMultiRepair: async (_id, value) => { calls.repair++; return value; },
        acquireAutomaticResearch: async (_id, value) => { calls.research++; return value; },
        certifyDeterministicResearch: noop,
        runRequestedTest: async () => { calls.test++; return {}; },
        patchHygieneIssue: async () => undefined, certifyCodeChange: noop,
        finalizeCodeChangeEvidence: noop, submitAutomaticCompletion: noop,
      });
      await complete['experimental.text.complete']({sessionID: 's'}, {text: 'done'});
      console.log(JSON.stringify({calls, prompt: prompt.system}));
    """
    result = _node(script, _profile("verification_only"))
    assert result == {
        "calls": {"repair": 0, "test": 0, "research": 0, "prompt": 0},
        "prompt": ["base"],
    }


def test_opencode_stopping_ablation_does_not_block_redundant_read() -> None:
    script = r"""
      import {createToolBeforeHook} from './src/cortheon/opencode_core/hook_tool_before.js';
      import {investigations} from './src/cortheon/opencode_core/state.js';
      investigations.set('s', {
        automatic: true, active: true, goal: 'Inspect docs',
        deliverable: 'document_synthesis', evidenceIDs: ['ev1'],
        plan: {operation: 'semantic_join', paths: ['a.md', 'b.md']},
      });
      const hook = createToolBeforeHook({
        debug: async () => {}, directory: '/repo', latestUserTask: async () => '',
        acquireRequestedEvidence: async () => false,
        captureMutationBefore: async () => {},
      });
      const output = {args: {filePath: 'a.md'}};
      await hook['tool.execute.before']({sessionID: 's', tool: 'read'}, output);
      console.log(JSON.stringify(output));
    """
    result = _node(script, _profile("without_adaptive_stopping"))
    assert result == {"args": {"filePath": "a.md"}}


def test_pi_stopping_ablation_removes_utility_stop_but_not_hard_cap(tmp_path) -> None:
    script = r"""
      import {discoveryExhausted, toolBudgetExhausted} from './pi_core/budget.ts';
      const active = {
        completed: false, deliverable: 'document_synthesis', request: undefined,
        goal: 'Find the cause by connecting clues across two documents.',
        evidenceRecords: [
          {id: 'ev1', source: 'pi:read:a.md'},
          {id: 'ev2', source: 'pi:read:b.md'},
        ],
        redundantDiscoveryCalls: 2,
      };
      console.log(JSON.stringify({
        adaptiveStop: discoveryExhausted(active),
        hardStop: toolBudgetExhausted({deliverable: 'answer', admittedToolCalls: 16}),
      }));
    """
    staged = _stage_pi(tmp_path)
    full = _node(script, _profile("full"), strip_types=True, cwd=staged)
    ablated = _node(
        script,
        _profile("without_adaptive_stopping"),
        strip_types=True,
        cwd=staged,
    )
    assert full == {"adaptiveStop": True, "hardStop": True}
    assert ablated == {"adaptiveStop": False, "hardStop": True}


def test_pi_and_opencode_apply_the_same_operator_flags() -> None:
    def flags(module: str) -> dict:
        script = f"""
      import {{operatorEnabled}} from '{module}';
      const keys = ['verification', 'cross_source_derivation'];
      console.log(JSON.stringify(Object.fromEntries(keys.map((key) => [key, operatorEnabled(key)]))));
    """
        return _node(
            script,
            _profile("without_cross_source_derivation"),
            strip_types=module.endswith(".ts"),
        )

    expected = {"verification": True, "cross_source_derivation": False}
    assert flags("./src/cortheon/pi_core/protocol.ts") == expected
    assert flags("./src/cortheon/opencode_core/state.js") == expected


def test_ci_python_matrix_installs_the_node_runtime_used_by_adapter_tests() -> None:
    workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    tests_job = workflow.split("  tests:\n", 1)[1]
    assert "actions/setup-node@v4" in tests_job
    assert 'node-version: "24"' in tests_job
