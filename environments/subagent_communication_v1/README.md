# Subagent Communication V1

This environment teaches the native Prime Agent depth-one RLM protocol. It does not
replace `rlm` or `agent_message` with environment tools: the policy executes the real
Prime Agent APIs in its persistent IPython kernel.

The four families separate capabilities that otherwise collapse into one sparse
success signal:

- `direct` keeps small work local and penalizes reflexive delegation.
- `single` retains one admission handle and receives one explicit child reply.
- `parallel` fans out two independent children and associates replies by stable name.
- `followup` withholds one parameter at spawn time and requires a parent-to-child
  follow-up before the child's final parent reply.

The specialist-population families expose a typed public capability registry and
keep the routing choice model-authored. `specialist_local` measures the decision
to retain owned work, `specialist_generic` provides a neutral terminal-worker
control, table join/reconciliation and source AST/config pairs define two worker
niches, and the recursive table/source families require a root-to-manager-to-worker
path. `available_experts` controls which registered workers are visible; it must
always include `generic_worker`. The environment records the latent preferred
expert for paired evaluation but does not reveal or enforce that choice.

Generator variants 0-3 train and variants 4-5 are held out. Final JSON correctness
has weight 1.0. Native protocol alignment has weight 0.35 and is derived from actual
IPython calls: callable `rlm`, retained handles, stable child names, role-addressed
messages with successful live receipts, delegated file paths, and non-repeated cells.
Set the Prime Agent harness to `max_depth = 1`;
depth zero disables the capability and depth two would test a different curriculum.
