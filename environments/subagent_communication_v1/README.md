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

Generator variants 0-3 train and variants 4-5 are held out. Final JSON correctness
has weight 1.0. Native protocol alignment has weight 0.35 and is derived from actual
IPython calls: callable `rlm`, retained handles, stable child names, role-addressed
messages, and non-repeated cells. Set the Prime Agent harness to `max_depth = 1`;
depth zero disables the capability and depth two would test a different curriculum.
