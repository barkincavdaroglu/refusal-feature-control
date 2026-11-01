# Protocol basis for schematic figures

For a source feature s, the edit adds Δh = Δa · dₛ to the original source-layer MLP output at the final prompt token. Here dₛ is the feature’s decoder direction. The rest of the original model then runs forward. The receiver measurement is its preactivation change Δz, before its ReLU.

AtP predicts Δz by the dot product of the original model’s receiver gradient with Δh. The graph prediction scales the recorded signed edge by Δa/a_ref, where a_ref is the source’s reference activation. Both predict the same finite edit and the same receiver. AtP uses the model’s computation but does not require the explicit attribution-graph representation.

In a separate analysis of the saved graphs, I tested whether the wiring changed feature selection. A backward walk starts at the target and follows incoming edges in proportion to their absolute weights. A feature’s influence is the probability that the walk visits it. A set’s coverage is the probability that the walk visits any member of the set, counting shared paths once. Greedy selection repeatedly adds the feature with the largest additional coverage; the baseline simply takes the highest individual influences.

What survived was that editing C reliably moved the internal scores. I therefore tested how much of that effect depended on the **whole layer-32 MLP output at the final prompt token**. The nominated individual feature in that layer remained a diagnostic; it was never the restored variable. Two endpoints were measured: refusal-start next-token log odds and the frozen probe.

T = D + B follows from the definitions. Checking it tests implementation consistency; the identity does not establish a unique path or mediation decomposition. Rescue is a separate intervention at the selected module boundary.

Development used six intents; replication used ten harmful and five benign intents disjoint from development, with natural per-prompt doses. Dependence and rescue each passed on 10/10 harmful intents. The route outcomes were unseen, though these prompts were not pristine with respect to all earlier project/model exposure. Benign C activations, and therefore their edits, were zero. A fair selectivity test needed the same nonzero edit on both sides.

A new panel used twelve harmful/benign topic pairs under one structured-planning template. These prompts had not previously been sent to the model in the project. Every prompt received +18.625 units of C, the harmful median dose from replication. A different layer-31 feature was scaled to the same decoded L2 norm as a comparator; the same edit one token earlier was a positional control. The panel ran once after freezing the rules.

Forced decoder addition bypasses the encoder and its ReLU threshold. Similar harmful and benign effects after forcing the same contribution do not show that C’s natural activation is unrelated to refusal. Selectivity could arise upstream, in whether that contribution is produced at all. Prior refusal-direction work likewise distinguishes extracting a naturally associated direction from testing an imposed direction [3]; the feature tested here is not established to be the same object.

The internal-route development, replication and fixed-dose seeds are 90701, 90801 and 90901. They use 223, 180 and 264 forwards, respectively, with no generated text. The comparator is layer 31, feature 139261. Check receiver hooks fire once; the one-token-earlier control leaves the final receiver and endpoints unchanged; and T = D + B holds numerically. These checks verify implementation consistency and timing, not mechanistic uniqueness.
