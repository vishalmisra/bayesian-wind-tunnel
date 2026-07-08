# Anthropic found a workspace inside Claude. We know what it is, and what it isn't.

*Companion post for our working note ["The Workspace Survives Where the
Circuit Dies: Routing Is Not
Computation"](https://github.com/vishalmisra/bayesian-wind-tunnel/blob/main/experiments/jlens/note/workspace-note.pdf)
(Dalal, Misra, Parekh; arXiv pending). Code and all experiments:
[bayesian-wind-tunnel](https://github.com/vishalmisra/bayesian-wind-tunnel).
Everything below reproduces in about two GPU-hours on one consumer
card.*

![The workspace is fine. We lost the writers.](assets/blog-header-cartoon.png)

A couple of days ago Anthropic published a striking result. Using a tool they call
the J-lens - Jacobians from internal activations to future output
logits - they found a small, densely connected subspace inside
production language models that holds reusable content the model
actually uses. They call it a global workspace. The finding set off a
wave of commentary, some of it about consciousness (we find below a
perfectly intact workspace in a model that has provably stopped
computing; make of that what you will), most of it missing the more
interesting question, which Anthropic themselves stated plainly: what
decides what enters the workspace, and how does it form?

Before anything else, what the J-lens actually is. Two pieces of
vocabulary.

First, the residual stream. Inside a transformer, every position in the
sequence carries its own vector, a kind of running scratchpad. Each
layer reads the scratchpads, computes something, and adds its output
back in. Attention heads are the components that read from other
positions' scratchpads and write into this one. Everything the model
knows at a given position, it knows because something wrote it there.

Second, "directions that matter." Take the scratchpad vector at some
layer and position and nudge it slightly in some direction. For most
directions, the model's future outputs do not move at all. For a few,
they move a lot. The J-lens computes this sensitivity exactly (it is a
matrix of derivatives, a Jacobian), and the J-space is the span of the
few directions the future actually depends on. That is the workspace:
not everything on the scratchpad, just the part the model goes on to
use. That is the lens's real innovation: it defines internal content
by use, not presence. A probe finds whatever is decodable, whether or
not the model ever reads it. The J-lens finds what the computation
runs through. Those are not the same thing, and we measure the
difference directly.

They cannot answer the formation question at production scale, and
neither can anyone else. A frontier model has no ground truth. You can find a
subspace, show it matters, and still not know what it *is*.

We have a setting where you can know. For the past year we have been
studying what we call [Bayesian wind
tunnels](https://arxiv.org/abs/2512.22471): tiny transformers, a few
million parameters, trained on tasks where the correct posterior is
known in closed form at every position. (For the accessible version of
that story, see [Attention Is Bayesian
Inference](https://medium.com/@vishalmisra/attention-is-bayesian-inference-578c25db4501).) The models become functionally
Bayesian - their uncertainty tracks the analytic posterior to within
thousandths of a bit - and because the task is small, we had already
mapped the mechanism: a hypothesis frame built in the first layer,
evidence eliminated through the middle, precision refined late.

So we ran Anthropic's lens inside the wind tunnel. If their workspace
is a real object and not an artifact of the method, the lens should
find it here. And here, unlike anywhere else, we can check what it
finds against the truth.

We wrote down five predictions with numeric thresholds before looking
at results. The note is candid about the dating: the protocol's public
commit postdates the first runs, so it is an internal prespecification,
not a timestamped preregistration. It also records where the gates held
strictly and where they needed refinement. Three of the five passed.
The two that failed were the two we would most have wanted to pass, and
they are the best results in the paper.

## What the workspace is

The identity checks came back stronger than we expected. The J-space
coincides with the hypothesis frame at roughly five to seven times the level of
a matched random subspace. Two controls pin this down: the same
architecture at random initialization shows nothing, and a trained
attention-free MLP shows nothing. The alignment is learned. Two more
controls, run at a referee's request, taught us what kind of fact it
is: a model trained with untied embeddings computes the same posterior
with much weaker token-frame alignment, and across training seeds the
alignment varies severalfold. The coordinates are a choice the
architecture and the training run make. What never varies, on any
seed, is the structure underneath: the contents decode perfectly, the
uncertainty direction is causally inert, and deleting the frame
destroys the model's calibration.

The contents check out too, but with a lesson attached. Linear probes
on the workspace coordinates read off exactly which hypotheses the
posterior has eliminated, at 99.8 percent balanced accuracy. (Balanced
matters: late in a sequence most hypotheses are already eliminated, so
a lazy always-"eliminated" probe scores 90 percent while learning
nothing; balanced accuracy makes lazy guessing score 50.) The lesson:
when a referee asked us to probe random subspaces of the same size, they
decoded almost as well. Decodable information is promiscuous - it leaks
into most fat subspaces of the residual stream - so a probe cannot tell
you where content lives. Which is our own point about use versus
presence, turned on us, and the reason the check that matters is
causal: build two sequences with identical evidence positions but
different hidden answers, swap the workspace content between them, and
the model's posterior follows the swapped evidence. Swap the frame and
you get complete redirection, indistinguishable from swapping the
entire residual stream. Swap a random subspace of the same size and
nothing ever happens - not once in a hundred draws. Probes say the
content is everywhere. Swaps say it is used from exactly one place.

One more thing rides on the workspace: a direction that encodes the
model's uncertainty, perfectly readable by a probe. Delete it and
calibration does not move. And the lens itself agrees: that direction's
projection onto the J-space sits at or below what a random direction
would have. The probe promotes it; the lens correctly leaves it out.
Readable is not the same as used. Keep that sentence; it comes back.

## The first failed prediction

We predicted a single dominant writer head, because our earlier work
had found one. A writer, in the scratchpad picture, is an attention
head whose output lands in the workspace directions rather than
elsewhere in the residual stream; a dominant writer would be one head
doing most of that writing, the way one librarian might maintain the
whole catalog. Wrong. On this model the workspace is written by a bank
of nine early heads, and ablating any of them hurts about equally.

But the failure resolved into something better. We retrained the exact
task variant from our earlier paper and the single-head result came
back. The difference is supervision geometry. Give the task one query
site and one binding head serves it. Make every position a readout
site and the model recruits a bank. Writer concentration tracks
readout demand. Both results are true; they are one phenomenon seen
from two training setups.

The failure also produced a practical warning for production-scale
work, and it is the paragraph I would most like practitioners to take
away. The obvious connectivity metric - the fraction of a head's
output that lands in the workspace - is gamed by nearly dead heads
whose tiny writes happen to point the right way. A head can write
almost nothing, have all of that nothing point at the workspace, and
top the ranking. Ranked by that ratio, the correlation with actual
importance comes out negative: the metric anti-selects. Ranked by
absolute projected write - how much workspace-directed output the head
actually produces - the nine heads that matter separate from the 27
that do not, with zero overlap between the groups. If you are
measuring writers in a big model, measure in absolute norm.

A speculative aside on LoRA, since the connection is hard to miss.
Low-rank adaptation works on the premise that task-relevant change
lives in a low-dimensional subspace. The wind tunnel is a ground-truth
case of the corresponding fact about activations: the subspace the
task runs through has dimension roughly equal to the number of
hypotheses, and our probes only saturate once their rank covers it.
That is consistent with why small LoRA ranks work at all, and it
suggests two testable corollaries. Adapters placed on writer layers
should move content while adapters placed late should move precision.
And any scheme that selects adapter sites by relative norms is exposed
to the same anti-selection failure we just measured.

## The second failed prediction, which is the headline

We have a family of models trained with a deliberate handicap: loss
only on the first five positions of a sequence. Our companion paper
shows these models fall off a cliff past position five - calibrated to
hundredths of a bit inside the horizon, off by more than a bit beyond
it.
Gradient descent compiled a local circuit, not a reusable mechanism.

We predicted the workspace would track the cliff: intact where the
model works, noise where it fails. The preregistration even said the
reverse outcome must be reported if found.

It was found. Past the horizon, where the model is essentially
guessing, the workspace geometry is still there, at nearly five times
the random baseline. What collapses is the computation that fills it.

Two interventions turn that dissociation into a mechanism. First, the
rescue: take the internal state that a working position would have
produced - modular arithmetic gives us exact state matching for free -
and transplant it into a broken position. Accuracy goes from 0.19 to
1.00. Everything downstream of the writers works fine, anywhere in the
sequence. The failure is confined to the machinery that writes.

Second, the converse. Rotate the part of the residual stream
orthogonal to the frame and prediction collapses to chance at every
depth. The frame tells a sharper story. Rotating it hurts exactly
where evidence enters: at block 1 the frame rotation cuts accuracy
roughly in half while a dimension-matched random rotation does
nothing. By block 3 both are harmless, so the sparing at depth is a
fact about dimension, not the frame. The frame is causally specific at
the entry band and generic after it. The in-flight computation lives
in the complement throughout. Transplants and corruptions agree: the
routing geometry and the computation that uses it are different
objects.

That is the sentence version of the paper. Routing is not computation.
A model can carry a perfectly good workspace into positions where
nothing knows how to write to it.

Why does the workspace survive at all? Because it is made of shared
weights. The token embeddings and attention maps that define the
geometry do not know what position they are at; position enters only
through how the weights get used. A positional loss mask can compile
position-dependence into the computation, never into the geometry.
Yes, this is just weight sharing. That is the point: the thesis of our
companion paper, visible through Anthropic's lens.

## When does it form?

We retrained the model saving checkpoints every hundred steps. The
frame crosses our detection threshold around step 400. We then reran
the causal swap at every checkpoint, and the workspace starts
redirecting the model's posterior between steps 400 and 500 - the same
steps it becomes detectable - while calibration is still a thousand
times worse than its final value. The remaining 47,000 steps buy
precision inside a substrate that already works. Structure first,
precision second. Our [gradient-dynamics
paper](https://arxiv.org/abs/2512.22473) predicted that ordering;
this is the causal version of it.

## Is any of this about transformers, or about intelligence?

The last section of the note is the part I keep thinking about. We ran
the whole pipeline on an LSTM and on a Mamba-style state-space model,
trained on the same task. The task only requires tracking which values
have been used up, so all three architectures learn it and all three
calibrate. Same Bayesian function, three organizations.

One structural difference drives everything in this section. A
transformer keeps a scratchpad at every position and lets attention
look back at all of them; the past stays spread out in space. A
recurrent network like an LSTM instead carries a running summary
forward one step at a time; the past is compressed into a state. Mamba,
a state-space model, is the modern version of the same idea: each
layer scans the sequence left to right, folding what it sees into a
small internal state as it goes. So the question "where does the model
keep what the evidence taught it" has structurally different possible
answers in each architecture.

The transformer keeps its workspace in the open: a shared token frame
in the residual stream, the same coordinates at every position. That
is what attention requires. Attention retrieves by matching content
across positions, and matching needs a common coordinate system. The
LSTM is frame-aligned only at the embedding, where evidence enters in
token coordinates. After that, every recurrent layer sits at exactly
null overlap with the frame - and yet the same layers carry the full
posterior in the LSTM's own private coordinates, a learned internal
code that looks nothing like the token directions. Probes decode it
perfectly, and swapping those private coordinates redirects the
posterior by four bits. Same content, same causal structure, different
coordinate system. Nothing forces an LSTM to keep the shared frame,
because nothing in it ever matches content across positions.

Mamba is stranger. No positional patch redirects it at all, not even
replacing the entire residual stream over the evidence region. That
sounds impossible until you remember the scan: by the time you patch
position twenty, every layer's internal state has already absorbed the
evidence while sweeping past it, and the patched scratchpads are not
where that history lives. The evidence is not at positions; it is
inside the scan states. Zero the scan states at the read boundary and
the model loses everything: KL to the true posterior explodes from
near zero to over 25 bits. The states carry the evidence.
Transplanting an evidence-matched donor's states redirects the
posterior on one of two seeds, so sufficiency is suggestive; necessity
is settled. The workspace exists there too, but it lives in state
space, where a positional residual-stream lens cannot reach it.

That last finding is a caution for the field. Run the J-lens on a
state-space or hybrid model and you may conclude there is no
workspace, while an intact, causally verified one sits where the lens
cannot look.

## The one-sentence version

Bayesian computation in sequence models factorizes into a routing
substrate and writer computation. The substrate is built early, as our
gradient-dynamics account predicted, is defined by shared weights so it
is global by construction, and is implemented in architecture-specific
coordinates: the transformer externalizes it into a shared frame, the
LSTM internalizes it into a private code, Mamba moves it into scan
state. The writer computation is what training actually has to buy,
position by position - and it, not the substrate, is where
generalization lives. Anthropic's J-lens gives us a way to measure the
substrate separately from the computation that uses it; run where the
ground truth is known, it shows the two are separable with a scalpel.
The separation is general. The coordinates are architectural.

And the corollary that matters beyond interpretability: having a
workspace and being able to use it are different properties, requiring
different evidence. Readable is not the same as used; neither is
present the same as functional. A model can fail to generalize with its substrate
fully intact, because generalization lives in the writers. If that
transfers to scale - and everything here is a 2.7M-parameter
existence proof, so treat it as a hypothesis - then questions about
what a big model "can represent" and what it "has learned to compute"
need to be asked, and tested, separately.

The note, the preregistration with both failures, the deviations log,
and every experiment are in the repo. It all runs in an afternoon on
one GPU. Most of it was, in fact, done on a flight from Hawaii to New
York. That was rather the point.

---

**Further reading.**

- [Attention Is Bayesian Inference](https://medium.com/@vishalmisra/attention-is-bayesian-inference-578c25db4501) - the accessible introduction to the program
- [The Bayesian Geometry of Transformer Attention](https://arxiv.org/abs/2512.22471) - Paper I: the wind tunnels, the hypothesis frame, which architectures can do exact Bayesian inference
- [Gradient Dynamics of Attention: How Cross-Entropy Sculpts Bayesian Manifolds](https://arxiv.org/abs/2512.22473) - Paper II: how training builds the geometry
- [Geometric Scaling of Bayesian Inference in LLMs](https://arxiv.org/abs/2512.23752) - Paper III: the same geometry in Pythia, Phi, Llama, and Mistral
- [bayesian-wind-tunnel](https://github.com/vishalmisra/bayesian-wind-tunnel) - code for everything above, including this note's experiments in `experiments/jlens/`
