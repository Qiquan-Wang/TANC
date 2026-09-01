/* TANC Visual Builder — app logic + Python code generator.
   No dependencies. Generates code for the `tanc` package. */
"use strict";

// ─────────────────────────────────────────────────────────────
// Catalogs
// ─────────────────────────────────────────────────────────────
// role: "auto" = derived from the previous layer's output (read-only when known);
//       "autodef" = a smart default (e.g. same-conv padding) that stays editable.
// gen   → PyTorch (nn.*) expression;  genTF(p, name) → Keras layer (or null if unsupported).
const _kpad = p => (_int(p.padding) === Math.floor((_int(p.kernel)||1)/2) && (_int(p.stride)||1) === 1) ? '"same"' : '"valid"';
const LAYERS = {
  linear:      { label: "Linear", prefix: "fc", weights: true,
                 params: [{k:"in", d:784, role:"auto"}, {k:"out", d:128}],
                 gen: p => `nn.Linear(${p.in}, ${p.out})`,
                 genTF: (p,n) => `layers.Dense(${p.out}, name=${n})` },
  conv2d:      { label: "Conv2d", prefix: "conv", weights: true,
                 params: [{k:"in", d:1, role:"auto"}, {k:"out", d:16}, {k:"kernel", d:3}, {k:"stride", d:1}, {k:"padding", d:1, role:"autodef"}],
                 gen: p => `nn.Conv2d(${p.in}, ${p.out}, kernel_size=${p.kernel}, stride=${p.stride}, padding=${p.padding})`,
                 genTF: (p,n) => `layers.Conv2D(${p.out}, ${p.kernel}, strides=${p.stride}, padding=${_kpad(p)}, name=${n})` },
  relu:        { label: "ReLU", prefix: "relu", weights: false, params: [],
                 gen: () => `nn.ReLU()`, genTF: (p,n) => `layers.ReLU(name=${n})` },
  maxpool2d:   { label: "MaxPool2d", prefix: "pool", weights: false,
                 params: [{k:"kernel", d:2}], gen: p => `nn.MaxPool2d(${p.kernel})`,
                 genTF: (p,n) => `layers.MaxPooling2D(pool_size=${p.kernel}, name=${n})` },
  flatten:     { label: "Flatten", prefix: "flat", weights: false, params: [],
                 gen: () => `nn.Flatten()`, genTF: (p,n) => `layers.Flatten(name=${n})` },
  batchnorm2d: { label: "BatchNorm2d", prefix: "bn", weights: false,
                 params: [{k:"features", d:16, role:"auto"}], gen: p => `nn.BatchNorm2d(${p.features})`,
                 genTF: (p,n) => `layers.BatchNormalization(name=${n})` },
  batchnorm1d: { label: "BatchNorm1d", prefix: "bn", weights: false,
                 params: [{k:"features", d:128, role:"auto"}], gen: p => `nn.BatchNorm1d(${p.features})`,
                 genTF: (p,n) => `layers.BatchNormalization(name=${n})` },
  dropout:     { label: "Dropout", prefix: "drop", weights: false,
                 params: [{k:"p", d:0.2, wide:true}], gen: p => `nn.Dropout(${p.p})`,
                 genTF: (p,n) => `layers.Dropout(${p.p}, name=${n})` },
  embedding:   { label: "Embedding", prefix: "emb", weights: true,
                 params: [{k:"num", d:1000}, {k:"dim", d:64}], gen: p => `nn.Embedding(${p.num}, ${p.dim})`,
                 genTF: (p,n) => `layers.Embedding(${p.num}, ${p.dim}, name=${n})` },
  soro:        { label: "SoRO", prefix: "soro", weights: true, needsSoRO: true,
                 params: [{k:"in", d:128, role:"auto"}, {k:"out", d:10}, {k:"rank", d:8}],
                 gen: p => `SoRO(${p.in}, ${p.out}, rank=${p.rank})`,
                 genTF: (p,n) => `SoRODense(${p.out}, ${p.rank}, name=${n})` },
  transformer: { label: "Transformer", prefix: "tenc", weights: true, seq: true,
                 params: [{k:"d_model", d:128, wide:true, role:"auto"}, {k:"nhead", d:4}, {k:"ff", d:256}],
                 gen: p => `nn.TransformerEncoderLayer(${p.d_model}, ${p.nhead}, dim_feedforward=${p.ff}, batch_first=True)`,
                 genTF: null },   // no single-layer Keras equivalent — PyTorch only
};

const DATASETS = {
  MNIST:        { ch:1, hw:28, classes:10 },
  FashionMNIST: { ch:1, hw:28, classes:10 },
  KMNIST:       { ch:1, hw:28, classes:10 },
  CIFAR10:      { ch:3, hw:32, classes:10 },
  CIFAR100:     { ch:3, hw:32, classes:100 },
};

// preset: builder/tool/traj + how to feed it + optional caveat
const PRESETS = [
  {k:"watanabe2021", label:"Watanabe & Yamana (2021) — weight PH (directed clique)", rep:"weights"},
  {k:"rieck2019",    label:"Rieck et al. (2019) — Neural Persistence (weight H0)", rep:"weights"},
  {k:"gebhart2019",  label:"Gebhart et al. (2019) — coupled weight PH", rep:"coupled"},
  {k:"lacombe2021",  label:"Lacombe et al. (2021) — Topological Uncertainty", rep:"coupled"},
  {k:"naitzat2020",  label:"Naitzat et al. (2020) — activation PH (geodesic)", rep:"activations"},
  {k:"karuppiah2025",label:"Karuppiah et al. (2025) — activation PH", rep:"activations"},
  {k:"ballester2024",label:"Ballester et al. (2024) — activation PH (ASDSQ)", rep:"activations",
     note:"Expects a single activation matrix; feed one recorded layer."},
  {k:"ramamurthy2019",label:"Ramamurthy et al. (2019) — labelled-complex PH", rep:"inputs_labels"},
  {k:"liu2023",      label:"Liu et al. (2023) — polyhedral PH", rep:"activations",
     note:"Uses ReLU activation patterns; results depend on a piecewise-linear net."},
  {k:"rathore2021",  label:"Rathore et al. (2021) — TopoAct (Mapper)", rep:"activations"},
  {k:"zhou2023",     label:"Zhou et al. (2023) — Mapper comparison", rep:"activations"},
  {k:"gabrielsson2019",label:"Gabrielsson & Carlsson (2019) — conv-kernel Mapper", rep:"weights",
     note:"Designed for conv kernels; include Conv2d layers."},
  {k:"gabella2021",  label:"Gabella (2021) — weight-trajectory Mapper", traj:true},
  {k:"ruppik2025",   label:"Ruppik et al. (2025) — local intrinsic dimension", rep:"activations"},
  {k:"ong2026",      label:"Ong et al. (2026) — calibrated intrinsic dimension", rep:"activations"},
  {k:"birdal2021",   label:"Birdal et al. (2021) — PH fractal dim of trajectory", traj:true},
  {k:"dupuis2023",   label:"Dupuis et al. (2023) — loss-PH dim of trajectory", traj:true, loss:true},
  {k:"andreeva2024", label:"Andreeva et al. (2024) — magnitude dim of trajectory", traj:true},
];

// What each preset needs from the rest of the configuration — used to warn the
// user about obviously-incompatible setups before they hit a runtime error.
//   act/weights: that aspect must be recorded · multi: needs Instances ≥ 2
//   conv3: needs a 3×3 Conv2d layer · longTraj: assumes a long training trajectory
const PRESET_NEEDS = {
  gebhart2019:    {act:true, weights:true}, lacombe2021: {act:true, weights:true},
  naitzat2020:    {act:true}, karuppiah2025:{act:true}, ballester2024:{act:true},
  ramamurthy2019: {act:true}, liu2023:{act:true}, rathore2021:{act:true},
  zhou2023:       {act:true}, ruppik2025:{act:true}, ong2026:{act:true},
  gabrielsson2019:{multi:true, conv3:true}, gabella2021:{multi:true},
  birdal2021:     {longTraj:true}, dupuis2023:{longTraj:true}, andreeva2024:{longTraj:true},
};

const PLOT_KINDS = {
  ph: ["diagram", "barcode", "betti_curve"],
  mapper: ["graph", "ph_diagram"],
  dimension: ["id_layers", "ph_scaling", "magnitude_scaling"],
};

// kwarg field schemas for the custom pipeline
const TOOL_KW = {
  ph: [
    {k:"max_dim", label:"max_dim", type:"select", opts:["0","1","2"], d:"1", py:v=>+v},
    {k:"backend", label:"backend (giotto = giotto-ph, parallel ripser — not giotto-tda)", type:"select",
     opts:["ripser","gudhi","giotto"], d:"ripser", py:v=>`"${v}"`},
    {k:"input_complex", label:"input_complex", type:"select", opts:["auto","directed_clique"], d:"auto", py:v=>`"${v}"`},
  ],
  mapper: [
    {k:"filter_fn", label:"filter_fn", type:"select", opts:["pca","l2_norm","eccentricity","entropy"], d:"pca", py:v=>`"${v}"`},
    {k:"n_intervals", label:"n_intervals", type:"num", d:"10", py:v=>+v},
    {k:"overlap_frac", label:"overlap_frac", type:"num", d:"0.3", py:v=>+v},
    {k:"n_components", label:"n_components", type:"num", d:"2", py:v=>+v},
    {k:"clusterer", label:"clusterer (agglomerative = single-linkage, Gabrielsson 2019)", type:"select",
     opts:["dbscan","agglomerative","histogram_gap"], d:"dbscan", py:v=>`"${v}"`},
  ],
  dimension: [
    {k:"estimator", label:"estimator", type:"select", opts:["activation_id","trajectory_dimension"], d:"activation_id", py:v=>`"${v}"`},
  ],
};
const DIM_METHOD = {
  activation_id: ["global","local","calibrated"],
  trajectory_dimension: ["magnitude","ph_euclidean","ph_loss"],
};
// "Over training" trajectory plots (tanc.visualisation) — computed per epoch.
//
// PH-based tracks (`construction: true`) share the OT_CONSTRUCTION fields: what the
// nodes/points of the per-epoch complex are — the default multipartite weight graph
// (with edge_weight choice), a point cloud of one weight matrix's rows/columns, or an
// activation cloud (samples or individual neurons as points). otConstruction() turns
// the chosen values into a `builder=` lambda in the generated code.
const OT_CONSTRUCTION = [
  {k:"cons", label:"nodes / points (construction)", type:"select", rerender:true,
   opts:["weight_graph","point_cloud (weights)","activation cloud"], d:"weight_graph"},
  // — weight_graph —
  {k:"wg_layer", label:"layer", type:"layer", allowAll:true, showIf:{k:"cons", in:["weight_graph"]}},
  {k:"edge_weight", label:"edge_weight", type:"select", showIf:{k:"cons", in:["weight_graph"]},
   opts:["normalized","absolute","global_normalized","relevance"], d:"normalized"},
  {k:"graph_scope", label:"graph_scope", type:"select", showIf:{k:"cons", in:["weight_graph"]},
   opts:["multipartite","bipartite","full"], d:"multipartite"},
  {k:"sparse", label:"sparse matrix (skip intra-layer non-edges — much faster H1, same diagram)",
   type:"select", opts:["no","yes"], d:"no", showIf:{k:"cons", in:["weight_graph"]}},
  // — point cloud of one weight matrix —
  {k:"pc_layer", label:"layer (one weight matrix)", type:"layer", showIf:{k:"cons", in:["point_cloud (weights)"]}},
  {k:"orientation", label:"orientation (rows = inputs, cols = neurons)", type:"select",
   opts:["rows","cols"], d:"cols", showIf:{k:"cons", in:["point_cloud (weights)"]}},
  {k:"pc_metric", label:"metric", type:"select", opts:["euclidean","cosine","correlation","cityblock"],
   d:"euclidean", showIf:{k:"cons", in:["point_cloud (weights)"]}},
  // — activation cloud —
  {k:"ac_layer", label:"layer (activations)", type:"layer", showIf:{k:"cons", in:["activation cloud"]}},
  {k:"points", label:"points", type:"select", opts:["samples (rows)","neurons (cols)"],
   d:"samples (rows)", showIf:{k:"cons", in:["activation cloud"]}},
  {k:"ac_distance", label:"distance", type:"select", opts:["euclidean","correlation","vne"],
   d:"euclidean", showIf:{k:"cons", in:["activation cloud"]}},
];

// The chosen construction → {pre: python lines defining _ot_builder, kw: extra call args, aspects}
function otConstruction(v) {
  const cons = v.cons || "weight_graph";
  if (cons === "point_cloud (weights)") {
    const L = v.pc_layer || "fc1";
    return {
      aspects: ["weights"],
      pre: [
        `from tanc.graph_builder import build_point_cloud_graph`,
        `_ot_builder = lambda snap: build_point_cloud_graph(snap.weights[${pyStr(L)}], orientation=${pyStr(v.orientation || "cols")}, metric=${pyStr(v.pc_metric || "euclidean")})`,
      ],
      kw: `, builder=_ot_builder`,
    };
  }
  if (cons === "activation cloud") {
    const L = v.ac_layer || "";
    const T = (v.points || "").startsWith("neurons") ? ".T" : "";
    return {
      aspects: ["activations"],
      pre: [
        `from tanc.graph_builder import build_activation_graph`,
        `_ot_builder = lambda snap: build_activation_graph(snap.activations[${pyStr(L)}]${T}, distance=${pyStr(v.ac_distance || "euclidean")})`,
      ],
      kw: `, builder=_ot_builder`,
    };
  }
  // weight_graph
  const allLayers = !v.wg_layer || v.wg_layer === "__all__";
  const sp = v.sparse === "yes" ? ", sparse=True" : "";   // sparse similarity graph → skip the dense flag complex
  const defaults = (v.edge_weight || "normalized") === "normalized"
                && (v.graph_scope || "multipartite") === "multipartite";
  if (defaults) {
    // the plot functions' own default construction — pass layer= only when one is chosen
    return { aspects: ["weights"], pre: [],
             kw: (allLayers ? "" : `, layer=${pyStr(v.wg_layer)}`) + sp };
  }
  const src = allLayers ? `list(snap.weights.values())` : `[snap.weights[${pyStr(v.wg_layer)}]]`;
  return {
    aspects: ["weights"],
    pre: [
      `from tanc.graph_builder import build_weight_graph`,
      `_ot_builder = lambda snap: build_weight_graph(${src}, edge_weight=${pyStr(v.edge_weight)}, graph_scope=${pyStr(v.graph_scope)})`,
    ],
    kw: `, builder=_ot_builder` + sp,
  };
}

const OVERTRAIN = {
  diagram_distance: {
    label: "Diagram distance to a reference epoch (Wasserstein / bottleneck)",
    fn: "plot_diagram_distance_trajectory", construction: true,
    fields: [
      {k:"ref",    label:"reference epoch (previous = per-epoch churn)", type:"select", opts:["previous","initial","final"], d:"previous"},
      {k:"metric", label:"metric",          type:"select", opts:["wasserstein","bottleneck"], d:"wasserstein"},
      {k:"dim",    label:"homology dim",     type:"select", opts:["0","1"], d:"1"},
      {k:"norm",   label:"scale-normalise (shape only)", type:"select", opts:["no","yes"], d:"no"},
      {k:"minpers",label:"prune persistence < (denoise + speed up distance; blank = keep all)", type:"num", d:""},
      {k:"settle", label:"mark settle epoch", type:"select", opts:["no","yes"], d:"no"},
    ],
    call: v => `view, ref=${pyStr(v.ref)}, metric=${pyStr(v.metric)}, dim=${+v.dim}`
      + (v.norm === "yes" ? `, normalize=True` : ``)
      + (v.minpers !== "" && v.minpers != null ? `, min_persistence=${+v.minpers}` : ``)
      + (v.settle === "yes" ? `, mark_settle=True` : ``),
  },
  ph_statistic: {
    label: "PH statistic over epochs (total persistence, entropy, …)",
    fn: "plot_ph_statistic_trajectory", construction: true,
    fields: [
      {k:"stat", label:"statistic", type:"select", opts:["total_persistence","persistence_norm","persistence_entropy","convex_hull_area"], d:"total_persistence"},
      {k:"dim",  label:"homology dim", type:"select", opts:["0","1"], d:"1"},
      {k:"norm",   label:"scale-normalise (shape only)", type:"select", opts:["no","yes"], d:"no"},
      {k:"settle", label:"mark settle epoch", type:"select", opts:["no","yes"], d:"no"},
    ],
    call: v => `view, stat=${pyStr(v.stat)}, dim=${+v.dim}`
      + (v.norm === "yes" ? `, normalize=True` : ``)
      + (v.settle === "yes" ? `, mark_settle=True` : ``),
  },
  betti: {
    label: "Betti number over epochs",
    fn: "plot_betti_trajectory", construction: true,
    fields: [
      {k:"dim", label:"homology dim", type:"select", opts:["0","1","2"], d:"1"},
      {k:"eps", label:"epsilon (blank = median birth per epoch)", type:"num", d:""},
    ],
    call: v => `view, dim=${+v.dim}` + (v.eps !== "" && v.eps != null ? `, epsilon=${+v.eps}` : ``),
  },
  dist_matrix: {
    label: "Diagram distance matrix (epoch × epoch heatmap)",
    fn: "plot_diagram_distance_matrix", construction: true,
    fields: [
      {k:"metric", label:"metric", type:"select", opts:["wasserstein","bottleneck"], d:"wasserstein"},
      {k:"dim",    label:"homology dim", type:"select", opts:["0","1"], d:"1"},
      {k:"minpers",label:"prune persistence < (denoise + speed up distance; blank = keep all)", type:"num", d:""},
    ],
    call: v => `view, metric=${pyStr(v.metric)}, dim=${+v.dim}`
      + (v.minpers !== "" && v.minpers != null ? `, min_persistence=${+v.minpers}` : ``),
  },
  pairplot: {
    label: "PH-statistic pairplot (trajectory through statistic space)",
    fn: "plot_ph_statistic_pairplot", construction: true,
    fields: [ {k:"dim", label:"homology dim", type:"select", opts:["0","1"], d:"1"} ],
    call: v => `view, dim=${+v.dim}`,
  },
  ph_panel: {
    label: "PH statistics panel (many diagram summaries over epochs)",
    fn: "plot_ph_statistics_panel", construction: true,
    fields: [
      {k:"dim",    label:"homology dim", type:"select", opts:["0","1"], d:"1"},
      {k:"settle", label:"mark settle epoch", type:"select", opts:["no","yes"], d:"no"},
    ],
    call: v => `view, dim=${+v.dim}`
      + (v.settle === "yes" ? `, mark_settle=True` : ``),
  },
  id: {
    label: "Intrinsic dimension of a layer over epochs",
    fn: "plot_id_over_training",
    aspectsFn: v => [v.source === "weights" ? "weights" : "activations"],
    fields: [
      {k:"layer",  label:"layer", type:"layer"},
      {k:"source", label:"point cloud from", type:"select", opts:["activations","weights"], d:"activations"},
      {k:"orientation", label:"points (activations: rows=samples, cols=neurons · weights: rows=inputs, cols=neurons)",
       type:"select", opts:["rows","cols"], d:"rows"},
      {k:"method", label:"method", type:"select", opts:["global","local"], d:"global"},
      {k:"settle", label:"mark settle epoch", type:"select", opts:["no","yes"], d:"no"},
    ],
    call: v => `view, ${pyStr(v.layer)}, methods=(${pyStr(v.method)},), source=${pyStr(v.source || "activations")}, orientation=${pyStr(v.orientation || "rows")}`
      + (v.settle === "yes" ? `, mark_settle=True` : ``),
  },
  id_all: {
    label: "Intrinsic dimension — all recorded layers (activations)",
    fn: "plot_id_trajectory_all_layers", aspects: ["activations"],
    fields: [
      {k:"method", label:"method", type:"select", opts:["global","local"], d:"global"},
      {k:"layout", label:"layout", type:"select", opts:["overlay","grid"], d:"overlay"},
    ],
    call: v => `view, method=${pyStr(v.method)}, layout=${pyStr(v.layout)}`,
  },
  phdim: {
    label: "PH fractal dimension of the weight trajectory (sliding window)",
    fn: "plot_ph_dimension_over_training", aspects: ["weights"],
    fields: [
      {k:"window", label:"window (epochs)", type:"num", d:"20"},
      {k:"stride", label:"stride", type:"num", d:"1"},
    ],
    call: v => `view, window=${+v.window || 20}, stride=${+v.stride || 1}`,
    validate: (v, epochs, warns) => {
      if (+epochs < (+v.window || 20))
        warns.push(`PH-dimension window (${+v.window || 20}) exceeds Epochs (${epochs}) — increase Epochs or shrink the window.`);
    },
  },
  magdim: {
    label: "Magnitude dimension of the weight trajectory (sliding window)",
    fn: "plot_magnitude_dimension_over_training", aspects: ["weights"],
    fields: [
      {k:"window", label:"window (epochs)", type:"num", d:"20"},
      {k:"stride", label:"stride", type:"num", d:"1"},
    ],
    call: v => `view, window=${+v.window || 20}, stride=${+v.stride || 1}`,
    validate: (v, epochs, warns) => {
      if (+epochs < (+v.window || 20))
        warns.push(`Magnitude-dimension window (${+v.window || 20}) exceeds Epochs (${epochs}) — increase Epochs or shrink the window.`);
    },
  },
};

const BUILDER_KW = {
  weight_graph: [
    {k:"edge_weight", label:"edge_weight (correlation / weighted_activation need coupled)", type:"select",
     opts:["normalized","absolute","global_normalized","relevance","correlation","weighted_activation"], d:"normalized", py:v=>`"${v}"`},
    {k:"graph_scope", label:"graph_scope", type:"select", opts:["multipartite","bipartite","full"], d:"multipartite", py:v=>`"${v}"`},
    {k:"induced_paths", label:"induced_paths (needs scope=full)", type:"select", opts:["no","yes"], d:"no", py:v=> v==="yes" ? "True" : null},
    {k:"node_feature_fn", label:"node_feature_fn (Mapper node features)", type:"select",
     opts:["laplacian_eigenvectors","degree_features","none"], d:"laplacian_eigenvectors",
     py:v=> v==="laplacian_eigenvectors" ? null : (v==="none" ? "None" : `"${v}"`)},
  ],
  point_cloud_graph: [
    {k:"orientation", label:"orientation (rows = inputs, cols = neurons)", type:"select", opts:["rows","cols"], d:"cols", py:v=>`"${v}"`},
    {k:"metric", label:"metric", type:"select", opts:["euclidean","cosine","correlation","cityblock"], d:"euclidean", py:v=>`"${v}"`},
  ],
  activation_graph: [
    {k:"distance", label:"distance", type:"select", opts:["euclidean","geodesic","correlation","vne"], d:"euclidean", py:v=>`"${v}"`},
    {k:"k", label:"k neighbours (geodesic; blank = auto)", type:"num", d:"", py:v=> v==="" ? null : +v},
    {k:"drop_constant", label:"drop constant neurons", type:"select", opts:["no","yes"], d:"no", py:v=> v==="yes" ? "True" : null},
    {k:"max_neurons", label:"max_neurons (blank = all)", type:"num", d:"", py:v=> v==="" ? null : +v},
  ],
  kernel_graph: [
    {k:"distance", label:"distance", type:"select", opts:["vne","euclidean"], d:"vne", py:v=>`"${v}"`},
    {k:"density_filter", label:"density_filter (drop sparse-region filters)", type:"select", opts:["no","yes"], d:"no", py:v=> v==="yes" ? "True" : null},
  ],
  labelled_complex_graph: [
    {k:"source_class", label:"source_class (blank = first label)", type:"num", d:"", py:v=> v==="" ? null : +v},
    {k:"gamma_quantile", label:"gamma_quantile (boundary proximity)", type:"num", d:"0.5", py:v=>+v},
  ],
  polyhedral_graph: [
    {k:"input_type", label:"input_type", type:"select", opts:["auto","activations","binary"], d:"auto",
     py:v=> v==="auto" ? null : `"${v}"`},
  ],
  None: [],
};

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────
const state = {
  layers: [],                 // {id, type, params:{}, name}
  analysisMode: "preset",
  framework: "torch",         // "torch" | "tf"
};
let _uid = 1;

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };

function autoName(type) {
  const pfx = LAYERS[type].prefix;
  let n = 1;
  const used = new Set(state.layers.map(l => l.name));
  while (used.has(pfx + n)) n++;
  return pfx + n;
}
function recordableLayers() { return state.layers.filter(l => LAYERS[l.type].weights); }
function hasSoRO() { return state.layers.some(l => l.type === "soro"); }
function hasSeq()  { return state.layers.some(l => LAYERS[l.type].seq); }

// ─────────────────────────────────────────────────────────────
// Shape inference — walk the stack and auto-fill input/padding/last-out.
// Shapes: {kind:"image",c,h,w} | {kind:"vector",n} | {kind:"seq",d} | null
// ─────────────────────────────────────────────────────────────
function datasetShape() {
  const d = DATASETS[$("#dataset").value];
  return d ? { kind:"image", c:d.ch, h:d.hw, w:d.hw } : null;   // null = custom/unknown
}
function flatSize(s) {
  if (!s) return null;
  if (s.kind === "image") return s.c * s.h * s.w;
  if (s.kind === "vector") return s.n;
  return null;
}
function shapeLabel(s) {
  if (!s) return "?";
  if (s.kind === "image")  return `${s.c}×${s.h}×${s.w}`;
  if (s.kind === "vector") return `${s.n}`;
  if (s.kind === "seq")    return `T×${s.d}`;
  return "?";
}
const _int = v => { const n = parseInt(v, 10); return Number.isFinite(n) ? n : 0; };

function traceShapes() {
  const classes = DATASETS[$("#dataset").value]?.classes || null;
  const paramLayers = state.layers.filter(l => LAYERS[l.type].weights);
  const lastFC = [...paramLayers].reverse().find(l => l.type === "linear" || l.type === "soro");
  let s = datasetShape();
  for (const L of state.layers) {
    if (!L.touched) L.touched = new Set();
    L._resolved = new Set();
    L._warn = null;
    L._in = s;
    s = applyLayer(L, s, { classes, isLastFC: L === lastFC });
    L._out = s;
  }
}
function applyLayer(L, s, ctx) {
  const p = L.params, t = L.type;
  const auto  = (k, v) => { p[k] = String(v); L._resolved.add(k); };            // read-only when set
  const adef  = (k, v) => { if (!L.touched.has(k)) { p[k] = String(v); L._resolved.add(k); } }; // editable default
  switch (t) {
    case "flatten":
      return s && s.kind === "image" ? { kind:"vector", n: flatSize(s) } : s;
    case "linear": case "soro": {
      const n = flatSize(s);
      if (n != null) auto("in", n);
      else if (s && s.kind !== "vector" && s.kind !== "image") L._warn = "expects a vector input";
      if (ctx.isLastFC && ctx.classes) adef("out", ctx.classes);
      return { kind:"vector", n: _int(p.out) };
    }
    case "conv2d": {
      if (s && s.kind === "image") auto("in", s.c);
      else L._warn = "expects an image (C×H×W) input";
      const k = _int(p.kernel) || 1, st = _int(p.stride) || 1;
      adef("padding", Math.floor(k / 2));                 // "same" for stride 1, odd kernel
      const pad = _int(p.padding);
      if (s && s.kind === "image") {
        const ho = Math.floor((s.h + 2*pad - k) / st) + 1;
        const wo = Math.floor((s.w + 2*pad - k) / st) + 1;
        if (ho < 1 || wo < 1) L._warn = "output spatial size < 1 — reduce kernel/stride";
        return { kind:"image", c:_int(p.out), h:Math.max(ho,0), w:Math.max(wo,0) };
      }
      return s;
    }
    case "maxpool2d": {
      const k = _int(p.kernel) || 2;
      if (s && s.kind === "image") return { kind:"image", c:s.c, h:Math.floor(s.h/k), w:Math.floor(s.w/k) };
      return s;
    }
    case "batchnorm2d": { if (s && s.kind === "image") auto("features", s.c); else L._warn = "expects image input"; return s; }
    case "batchnorm1d": { const n = flatSize(s); if (n != null) auto("features", n); return s; }
    case "relu": case "dropout": return s;
    case "embedding": return { kind:"seq", d:_int(p.dim) };
    case "transformer": {
      if (s && s.kind === "seq") auto("d_model", s.d);
      else L._warn = "expects a sequence (batch, seq, d_model) input";
      return s && s.kind === "seq" ? s : { kind:"seq", d:_int(p.d_model) };
    }
    default: return s;
  }
}

// ─────────────────────────────────────────────────────────────
// Render: palette + stack
// ─────────────────────────────────────────────────────────────
function renderPalette() {
  const p = $("#palette"); p.innerHTML = "";
  for (const [type, def] of Object.entries(LAYERS)) {
    const b = el("button", null, def.label);
    b.onclick = () => addLayer(type);
    p.appendChild(b);
  }
}

function addLayer(type) {
  const def = LAYERS[type];
  const params = {}; def.params.forEach(pp => params[pp.k] = pp.d);
  state.layers.push({ id: _uid++, type, params, name: autoName(type), touched: new Set() });
  renderAll(); regen();   // full re-render: the record checkbox list, rep-layer and
                          // over-training layer dropdowns must all see the new layer
}

function renderStack() {
  traceShapes();                       // auto-fill input/padding/last-out before rendering
  const stack = $("#layer-stack"); stack.innerHTML = "";
  $("#stack-empty").classList.toggle("hidden", state.layers.length > 0);
  state.layers.forEach((L, idx) => {
    const def = LAYERS[L.type];
    const li = el("li", "layer" + (def.weights ? " recordable" : ""));
    li.draggable = true; li.dataset.idx = idx;

    const drag = el("span", "drag", "⋮⋮");
    const info = el("div");
    const nameRow = el("div");
    const nameIn = el("input"); nameIn.value = L.name; nameIn.style.width = "84px";
    nameIn.className = "lname-in";
    nameIn.onchange = () => { L.name = nameIn.value.trim() || L.name; renderRecordExplicit(); renderRepLayer(); regen(); };
    const tt = el("span", "ltype", " " + def.label);
    // shape caption: in → out (and any warning)
    const shape = el("span", "shape", `  ${shapeLabel(L._in)} → ${shapeLabel(L._out)}`);
    nameRow.appendChild(nameIn); nameRow.appendChild(tt); nameRow.appendChild(shape);
    if (L._warn) { const w = el("span", "shape-warn", " ⚠ " + L._warn); nameRow.appendChild(w); }

    const params = el("div", "params");
    def.params.forEach(pp => {
      const isAuto = pp.role === "auto" && L._resolved.has(pp.k);
      const lab = el("label", isAuto ? "auto" : null, pp.k);
      const inp = el("input"); if (pp.wide) inp.className = "wide";
      inp.value = L.params[pp.k]; inp.type = "text";
      if (isAuto) { inp.readOnly = true; inp.title = "auto — inferred from the previous layer"; }
      else {
        inp.onchange = () => {
          L.params[pp.k] = inp.value.trim();
          L.touched.add(pp.k);              // stop auto-defaults from overwriting it
          renderStack(); regen();           // re-trace: downstream shapes may change
        };
      }
      lab.appendChild(inp); params.appendChild(lab);
    });
    info.appendChild(nameRow); info.appendChild(params);

    const actions = el("div", "layer-actions");
    const up = el("button", "ghost sm", "↑"); up.onclick = () => move(idx, -1);
    const dn = el("button", "ghost sm", "↓"); dn.onclick = () => move(idx, +1);
    const rm = el("button", "ghost sm danger", "✕"); rm.onclick = () => { state.layers.splice(idx,1); renderAll(); regen(); };
    actions.append(up, dn, rm);

    li.append(drag, info, actions);
    // drag & drop reorder
    li.addEventListener("dragstart", e => { li.classList.add("dragging"); e.dataTransfer.setData("text/plain", idx); });
    li.addEventListener("dragend", () => li.classList.remove("dragging"));
    li.addEventListener("dragover", e => e.preventDefault());
    li.addEventListener("drop", e => {
      e.preventDefault();
      const from = +e.dataTransfer.getData("text/plain");
      const to = idx; if (from === to) return;
      const [m] = state.layers.splice(from, 1); state.layers.splice(to, 0, m);
      renderAll(); regen();
    });
    stack.appendChild(li);
  });
}
function move(idx, d) {
  const j = idx + d; if (j < 0 || j >= state.layers.length) return;
  [state.layers[idx], state.layers[j]] = [state.layers[j], state.layers[idx]];
  renderAll(); regen();
}

// ─────────────────────────────────────────────────────────────
// Render: recording + analysis dependent bits
// ─────────────────────────────────────────────────────────────
function renderRecordExplicit() {
  const box = $("#record-explicit"); box.innerHTML = "";
  state.layers.forEach(L => {
    const def = LAYERS[L.type];
    const lab = el("label", "chk");
    const cb = el("input"); cb.type = "checkbox"; cb.value = L.name; cb.dataset.name = L.name;
    if (def.weights) cb.checked = true;
    cb.onchange = regen;
    lab.appendChild(cb); lab.appendChild(document.createTextNode(" " + L.name + " "));
    const t = el("span", "ltype", def.label); lab.appendChild(t);
    box.appendChild(lab);
  });
}
function renderRepLayer() {
  const sel = $("#rep-layer"); sel.innerHTML = "";
  recordableLayers().forEach(L => {
    const o = el("option"); o.value = L.name; o.textContent = L.name; sel.appendChild(o);
  });
}

function renderPresets() {
  const sel = $("#preset"); sel.innerHTML = "";
  PRESETS.forEach(p => { const o = el("option"); o.value = p.k; o.textContent = p.label; sel.appendChild(o); });
  updatePresetDesc();
}
function updatePresetDesc() {
  const p = PRESETS.find(x => x.k === $("#preset").value);
  $("#preset-desc").textContent = p?.note ? "Note: " + p.note : (p?.traj ? "Trajectory analysis — trains and captures snapshots across epochs." : "Single-snapshot analysis on the trained model.");
}

function renderToolKwargs() {
  const tool = $("#tool").value;
  const box = $("#tool-kwargs");
  const prevEst = getKw("tk", "estimator");        // preserve across the re-render
  box.innerHTML = "";
  (TOOL_KW[tool] || []).forEach(f => box.appendChild(kwField("tk", f)));
  if (tool === "dimension") {
    // restore the chosen estimator (kwField reset it to its default) so switching
    // to trajectory_dimension sticks and offers the right method options.
    const estSel = box.querySelector('[data-ns="tk"][data-kw="estimator"]');
    if (prevEst && estSel) estSel.value = prevEst;
    const est = (estSel && estSel.value) || "activation_id";
    const methodField = {k:"method", label:"method", type:"select", opts:DIM_METHOD[est], d:DIM_METHOD[est][0], py:v=>`"${v}"`};
    box.appendChild(kwField("tk", methodField));
  }
  renderPlotKinds();
}
function renderBuilderKwargs() {
  const b = $("#builder").value;
  const box = $("#builder-kwargs"); box.innerHTML = "";
  (BUILDER_KW[b] || []).forEach(f => box.appendChild(kwField("bk", f)));
  // sensible default representation for the builder
  const rep = (b === "weight_graph" || b === "point_cloud_graph") ? "weights"
            : b === "kernel_graph" ? "__kernels__"
            : b === "labelled_complex_graph" ? "inputs_labels"
            : "activations";
  $("#representation").value = rep;
  onRepChange();
}
function kwField(ns, f) {
  const lab = el("label", null, f.label);
  let inp;
  if (f.type === "select") {
    inp = el("select");
    f.opts.forEach(o => { const op = el("option"); op.value = o; op.textContent = o; inp.appendChild(op); });
    inp.value = f.d;
  } else {
    inp = el("input"); inp.type = "text"; inp.value = f.d;
  }
  inp.dataset.kw = f.k; inp.dataset.ns = ns; inp.dataset.py = "";
  inp._pyfn = f.py;
  inp.onchange = () => { if (ns === "tk" && f.k === "estimator") renderToolKwargs(); regen(); };
  lab.appendChild(inp);
  return lab;
}
function getKw(ns, k) {
  const inp = document.querySelector(`[data-ns="${ns}"][data-kw="${k}"]`);
  return inp ? inp.value : null;
}
function collectKwargs(ns) {
  const out = {};
  $$(`[data-ns="${ns}"]`).forEach(inp => {
    const v = inp._pyfn(inp.value);
    if (v !== null && v !== undefined && v !== "") out[inp.dataset.kw] = v;  // skip blank/optional
  });
  return out;
}
function renderPlotKinds() {
  const tool = $("#tool").value;
  const sel = $("#plot-kind"); const prev = sel.value; sel.innerHTML = "";
  (PLOT_KINDS[tool] || []).forEach(k => { const o = el("option"); o.value = k; o.textContent = k; sel.appendChild(o); });
  // Dimension tool: the right plot depends on the estimator/method, so default to
  // the matching one (id_layers is meaningless for a trajectory fractal dimension).
  if (tool === "dimension") {
    const est = getKw("tk", "estimator") || "activation_id";
    const method = getKw("tk", "method") || "";
    const want = est === "trajectory_dimension"
      ? (method === "magnitude" ? "magnitude_scaling" : "ph_scaling")
      : "id_layers";
    const prevOk = est === "trajectory_dimension"
      ? (prev === "ph_scaling" || prev === "magnitude_scaling")
      : prev === "id_layers";
    sel.value = prevOk ? prev : want;
    return;
  }
  if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
}
function onRepChange() {
  const v = $("#representation").value;
  $("#rep-layer-wrap").classList.toggle("hidden", v !== "__layer__" && v !== "__kernels__");
}

function renderOvertrainTrack() {
  const sel = $("#overtrain-track");
  if (!sel.options.length) {
    Object.entries(OVERTRAIN).forEach(([k, v]) => {
      const o = el("option"); o.value = k; o.textContent = v.label; sel.appendChild(o);
    });
  }
  renderOvertrainFields();
}
function overtrainFieldList(conf) {
  // PH-based tracks share the construction fields (what the nodes/points are)
  return conf.construction ? [...OT_CONSTRUCTION, ...conf.fields] : conf.fields;
}
function renderOvertrainFields() {
  const sel = $("#overtrain-track");
  if (!sel.options.length) return;
  const prev = collectOvertrain();                 // preserve current values across re-renders
  const box = $("#overtrain-fields"); box.innerHTML = "";
  const conf = OVERTRAIN[sel.value];
  const fields = overtrainFieldList(conf);
  const valueOf = k => {
    if (prev[k] != null) return prev[k];
    const f = fields.find(x => x.k === k);
    return f ? f.d : null;
  };
  fields.forEach(f => {
    if (f.showIf && !f.showIf.in.includes(valueOf(f.showIf.k))) return;   // hidden for this construction
    const lab = el("label", null, f.label);
    let inp;
    if (f.type === "layer") {
      inp = el("select");
      if (f.allowAll) { const o = el("option"); o.value = "__all__"; o.textContent = "(all recorded)"; inp.appendChild(o); }
      recordableLayers().forEach(L => { const o = el("option"); o.value = L.name; o.textContent = L.name; inp.appendChild(o); });
    } else if (f.type === "select") {
      inp = el("select");
      f.opts.forEach(o => { const op = el("option"); op.value = o; op.textContent = o; inp.appendChild(op); });
      inp.value = f.d;
    } else {
      inp = el("input"); inp.type = "text"; inp.value = f.d;
    }
    if (prev[f.k] != null && (!inp.options || [...inp.options].some(o => o.value === prev[f.k]))) inp.value = prev[f.k];
    inp.dataset.ot = f.k;
    inp.onchange = f.rerender ? () => { renderOvertrainFields(); regen(); } : regen;
    lab.appendChild(inp); box.appendChild(lab);
  });
}
function collectOvertrain() {
  const v = {};
  $$("#overtrain-fields [data-ot]").forEach(inp => { v[inp.dataset.ot] = inp.value; });
  return v;
}

function renderAll() { renderStack(); renderRecordExplicit(); renderSweepLayers(); renderRepLayer(); renderOvertrainFields(); updateShapeHint(); updatePoolingVisibility(); }

function updateShapeHint() {
  const d = DATASETS[$("#dataset").value];
  $("#shape-hint").textContent = d ? `input ${d.ch}×${d.hw}×${d.hw} · flattened = ${d.ch*d.hw*d.hw} · ${d.classes} classes` : "custom data";
}
function updatePoolingVisibility() {
  $("#pooling-wrap").classList.toggle("hidden", !(hasSeq() && $("#asp-activations").checked));
}

// ─────────────────────────────────────────────────────────────
// Code generation
// ─────────────────────────────────────────────────────────────
// ── Mapper sweep ────────────────────────────────────────────────────────────
// Values come from checkbox groups rather than free text: the option sets are
// finite, and a typo in a layer name would otherwise surface as a KeyError deep
// inside a run that had already trained a population.

// Layer checkboxes are built from the model actually assembled above, so the
// names cannot disagree with it.
function renderSweepLayers() {
  const box = $("#sweep-layers"); if (!box) return;
  const prev = new Set(checked("#sweep-layers"));
  box.innerHTML = "";
  const layers = recordableLayers();
  if (!layers.length) {
    box.innerHTML = '<p class="hint">Add a Linear, Conv or SoRO layer to the model above.</p>';
    return;
  }
  layers.forEach((L, i) => {
    const lab = el("label", "chk");
    const cb = el("input"); cb.type = "checkbox"; cb.dataset.v = L.name;
    cb.checked = prev.size ? prev.has(L.name) : i === 0;
    cb.onchange = regen;
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(" " + L.name + " "));
    lab.appendChild(el("span", "ltype", LAYERS[L.type].label));
    box.appendChild(lab);
  });
}

// Ticked values of a checkbox group, in DOM order.
function checked(sel) {
  return $$(sel + " input[type=checkbox][data-v]")
    .filter(c => c.checked).map(c => c.dataset.v);
}

// "other" free-text field, split and trimmed. Returns [values, badTokens].
function others(sel, validate) {
  const raw = ($(sel)?.value || "").split(",").map(x => x.trim()).filter(Boolean);
  const good = [], bad = [];
  raw.forEach(x => (validate(x) ? good : bad).push(x));
  return [good, bad];
}

const isInt   = x => /^\d+$/.test(x) && +x >= 1;
const isOvl   = x => /^\d*\.?\d+$/.test(x) && +x >= 0 && +x < 1;
const isPct   = x => /^\d*\.?\d+$/.test(x) && +x > 0 && +x <= 100;
const isFrac  = x => /^\d*\.?\d+$/.test(x) && +x > 0 && +x <= 1;

// A single value renders as a pinned axis; several render as a swept list.
function pyAxis(vals, quote) {
  const f = quote ? pyStr : (x => x);
  return vals.length === 1 ? f(vals[0]) : "[" + vals.map(f).join(", ") + "]";
}

function renderQuantileVisibility() {
  const k = $("#sweep-clusterer")?.value || "dbscan_q";
  const auto = (k === "first_gap" || k === "hdbscan" || k === "dbscan_elbow");
  const box = $("#sweep-quantiles-box");
  if (box) {
    box.classList.toggle("hidden", auto);
    const lbl = box.previousElementSibling;
    if (lbl && lbl.tagName === "LABEL") lbl.classList.toggle("hidden", auto);
  }
  $("#sweep-linkage-wrap")?.classList.toggle("hidden", k !== "first_gap");
  $("#sweep-mcs-wrap")?.classList.toggle("hidden", k !== "hdbscan");
}

function renderFilterStrength() {
  const kind = $("#sweep-filter")?.value || "none";
  const wrap = $("#sweep-filter-strength-wrap"), fld = $("#sweep-filter-strength"),
        hint = $("#sweep-filter-hint");
  if (!wrap) return;
  wrap.classList.toggle("hidden", kind === "none");
  if (kind === "none") { if (hint) hint.textContent = ""; return; }
  if (kind === "density") {
    if (fld && !fld.dataset.touched) fld.value = "1.0, 0.5, 0.3";
    if (hint) hint.textContent = "Fraction of points kept, in (0, 1]. 1.0 keeps everything; "
      + "0.3 keeps the densest 30%. Each value is swept, always alongside the unfiltered cloud.";
  } else {
    if (fld && !fld.dataset.touched) fld.value = "25, 50";
    if (hint) hint.textContent = "Percentile cut on kernel norm, in (0, 100]. "
      + "50 keeps the half above (or below) the median.";
  }
}

function collectSweep(warns) {
  const layers = checked("#sweep-layers");
  // Conv views live in their own group because they mean something different:
  // they name which axes of a 4-D weight index points. A name appearing in both
  // groups ("rows", "cols") is the same string in the generated code — the
  // toolkit dispatches on the weight's rank, not on which box was ticked.
  const views  = [...new Set(checked("#sweep-views").concat(checked("#sweep-conv-views")))];
  const lenses = checked("#sweep-lenses");
  const preps  = checked("#sweep-preps");

  const [iOther, iBad] = others("#sweep-intervals-other", isInt);
  const [oOther, oBad] = others("#sweep-overlap-other",   isOvl);
  const [qOther, qBad] = others("#sweep-quantiles-other", isPct);
  const intervals = checked("#sweep-intervals-box").concat(iOther);
  const overlaps  = checked("#sweep-overlap-box").concat(oOther);
  const quantiles = checked("#sweep-quantiles-box").concat(qOther);

  if (warns) {
    if (iBad.length) warns.push(`Intervals “${iBad.join(", ")}” ignored — resolution must be a whole number of cover intervals (1 or more).`);
    if (oBad.length) warns.push(`Overlap “${oBad.join(", ")}” ignored — overlap must satisfy 0 ≤ o < 1. Interval width diverges as it approaches 1.`);
    if (qBad.length) warns.push(`Quantile “${qBad.join(", ")}” ignored — a distance percentile must be greater than 0 and at most 100.`);
    if (!layers.length)    warns.push("Mapper sweep: tick at least one layer to analyse.");
    if (!views.length)     warns.push("Mapper sweep: tick at least one view.");
    if (!lenses.length)    warns.push("Mapper sweep: tick at least one lens.");
    if (!intervals.length) warns.push("Mapper sweep: tick at least one resolution, or type one in the “other” box.");
    if (!overlaps.length)  warns.push("Mapper sweep: tick at least one overlap, or type one in the “other” box.");
  }

  // Rescaling only -- specs that remove points belong on the filter axis, and
  // the library now rejects them here.  "none" is Python None.
  const prepVals = (preps.length ? preps : ["none"])
    .map(v => v === "none" ? "None" : pyStr(v));
  const prepExpr = prepVals.length === 1 ? prepVals[0] : "[" + prepVals.join(", ") + "]";

  // point filter: selects a subset, applied AFTER the lens with the cover pinned.
  const fkind = $("#sweep-filter")?.value || "none";
  let filtExpr = "None", nFilt = 1;
  if (fkind !== "none") {
    const ok = fkind === "density" ? isFrac : isPct;
    const [fv, fBad] = others("#sweep-filter-strength", ok);
    if (warns && fBad.length)
      warns.push(`Filter strength “${fBad.join(", ")}” ignored — ` + (fkind === "density"
        ? "a density keep-fraction must be in (0, 1]."
        : "a norm percentile must be in (0, 100]."));
    const items = fv.map(v => fkind === "density"
      ? `("density", 200, ${v})`
      : `("norm", ${v}, ${pyStr(fkind === "norm_high" ? "high" : "low")})`);
    if (items.length) {
      const all = ["None"].concat(items);
      filtExpr = "[" + all.join(", ") + "]"; nFilt = all.length;
    } else if (warns) {
      warns.push("Point filter selected but no valid strength given — running unfiltered.");
    }
  }

  // clusterer: the elbow rule takes no threshold, so it is a single setting.
  const clus = $("#sweep-clusterer")?.value || "dbscan_q";
  let clusExpr, nClus;
  if (clus === "first_gap") {
    // takes no threshold at all: the cut is chosen per cell from its own dendrogram
    const lk = $("#sweep-linkage")?.value || "single";
    clusExpr = `FirstGapCells(relative_gap=0.3, linkage=${pyStr(lk)})`; nClus = 1;
  } else if (clus === "hdbscan") {
    // density-based with no radius; min_cluster_size is a count, not a distance
    const mcs = Math.max(2, parseInt($("#sweep-mcs")?.value) || 5);
    clusExpr = `HDBSCANCells(min_cluster_size=${mcs})`; nClus = 1;
  } else if (clus === "dbscan_elbow") {
    clusExpr = `DBSCANCells(eps="elbow", min_samples=5)`; nClus = 1;
  } else {
    const cls = clus === "single" ? "SingleLinkageCells"
              : clus === "ward"   ? "WardCells" : "DBSCANCells";
    const arg = clus === "dbscan_q" ? "eps" : "threshold";
    const qs  = quantiles.length ? quantiles : ["5"];
    const items = qs.map(q => `${cls}(${arg}=("quantile", ${q}))`);
    clusExpr = items.length === 1 ? items[0]
             : "[" + items.join(",\n                    ") + "]";
    nClus = items.length;
  }

  const untrained = !!$("#sweep-untrained")?.checked;
  const nClouds = layers.length * views.length * (untrained ? 2 : 1);
  const nConfigs = nClouds * prepVals.length * lenses.length * nFilt
                 * intervals.length * overlaps.length * nClus;

  return { layers, views, lenses, intervals, overlaps, clusExpr, prepExpr,
           filtExpr, untrained, nClouds, nConfigs,
           prepCount: prepVals.length, filtCount: nFilt, clusCount: nClus };
}

function pyStr(s){ return `"${s}"`; }
function pyList(arr){ return "[" + arr.map(pyStr).join(", ") + "]"; }
function dictLiteral(obj){
  const items = Object.entries(obj).map(([k,v]) => `"${k}": ${v}`);
  return "{" + items.join(", ") + "}";
}

// Emits the `def _mapper_html(...)` helper (rotatable 3-D plotly Mapper graph).
function mapperHtmlHelperLines() {
  return [
    `def _mapper_html(r, path, color_by=None):     # rotatable 3-D Mapper graph (plotly)`,
    `    import networkx as nx, plotly.graph_objects as go`,
    `    G = r.mapper_graph; members = r.mapper_node_members or {}; fv = r.mapper_filter_values`,
    `    pos = nx.spring_layout(G, dim=3, seed=0); nodes = list(G.nodes())`,
    `    sizes = [len(members.get(n, [])) for n in nodes]; smax = max(sizes) or 1`,
    `    if color_by is not None:`,
    `        color = [float(np.mean(color_by[members[n]])) if members.get(n) else 0.0 for n in nodes]; cbar = "run"`,
    `    elif fv is not None:`,
    `        color = [float(np.mean(fv[members[n], 0])) if members.get(n) else 0.0 for n in nodes]; cbar = "filter"`,
    `    else:`,
    `        color = sizes; cbar = "size"`,
    `    ex, ey, ez = [], [], []`,
    `    for u, v in G.edges():`,
    `        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]; ez += [pos[u][2], pos[v][2], None]`,
    `    fig = go.Figure()`,
    `    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",`,
    `        line=dict(color="rgba(130,130,130,0.4)", width=2), hoverinfo="none"))`,
    `    fig.add_trace(go.Scatter3d(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], z=[pos[n][2] for n in nodes],`,
    `        mode="markers", text=[f"node {n}<br>{sizes[i]} samples" for i, n in enumerate(nodes)], hoverinfo="text",`,
    `        marker=dict(size=[8 + 22 * (s / smax) for s in sizes], color=color, colorscale="Viridis",`,
    `                    showscale=True, colorbar=dict(title=cbar), line=dict(width=0.5, color="#333"))))`,
    `    fig.update_layout(title="Mapper graph (drag to rotate)", showlegend=False, margin=dict(l=0, r=0, t=30, b=0),`,
    `        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))`,
    `    fig.write_html(path, include_plotlyjs=True, full_html=True)`,
  ];
}

// Results section: static PNG per result, or an interactive 3-D Mapper HTML.
function resultsLines(interactiveMapper, plotKind) {
  const L = [];
  L.push(`# ── Results ───────────────────────────────────────────────────────`);
  if (interactiveMapper) {
    mapperHtmlHelperLines().forEach(x => L.push(x));
    L.push(`_results = result if isinstance(result, list) else [result]`);
    L.push(`for i, r in enumerate(_results):`);
    L.push(`    _mapper_html(r, f"tda_result_{i}.html")`);
    L.push(`    print(f"saved interactive graph -> tda_result_{i}.html")`);
  } else {
    L.push(`_results = result if isinstance(result, list) else [result]`);
    L.push(`for i, r in enumerate(_results):`);
    if (plotKind) L.push(`    fig = r.plot(${pyStr(plotKind)}, save=f"tda_result_{i}.png")`);
    else L.push(`    fig = r.plot(save=f"tda_result_{i}.png")   # default plot for this analysis`);
    L.push(`    print(f"saved figure -> tda_result_{i}.png")`);
  }
  return L;
}

// Self-contained, torchvision-free dataset loader → numpy (N, C, H, W) in [0,1].
function datasetLoaderLines(dsName) {
  const L = [];
  L.push(`def _fetch(url, path):`);
  L.push(`    if not os.path.exists(path):`);
  L.push(`        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)`);
  L.push(`        print("downloading", url); urllib.request.urlretrieve(url, path)`);
  L.push(`    return path`);
  if (dsName === "CIFAR10" || dsName === "CIFAR100") {
    L.push(`def load_dataset(name, root="./data"):`);
    L.push(`    url = {"CIFAR10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",`);
    L.push(`           "CIFAR100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"}[name]`);
    L.push(`    tgz = _fetch(url, os.path.join(root, os.path.basename(url)))`);
    L.push(`    with tarfile.open(tgz) as t: t.extractall(root)`);
    L.push(`    def _b(f, key):`);
    L.push(`        with open(f, "rb") as fh: d = pickle.load(fh, encoding="bytes")`);
    L.push(`        return d[b"data"], np.array(d[key])`);
    L.push(`    if name == "CIFAR10":`);
    L.push(`        fold = os.path.join(root, "cifar-10-batches-py"); xs, ys = [], []`);
    L.push(`        for i in range(1, 6):`);
    L.push(`            x, y = _b(os.path.join(fold, f"data_batch_{i}"), b"labels"); xs.append(x); ys.append(y)`);
    L.push(`        Xtr = np.concatenate(xs); ytr = np.concatenate(ys)`);
    L.push(`        Xte, yte = _b(os.path.join(fold, "test_batch"), b"labels")`);
    L.push(`    else:`);
    L.push(`        fold = os.path.join(root, "cifar-100-python")`);
    L.push(`        Xtr, ytr = _b(os.path.join(fold, "train"), b"fine_labels")`);
    L.push(`        Xte, yte = _b(os.path.join(fold, "test"), b"fine_labels")`);
    L.push(`    Xtr = Xtr.reshape(-1, 3, 32, 32).astype("float32") / 255.0`);
    L.push(`    Xte = Xte.reshape(-1, 3, 32, 32).astype("float32") / 255.0`);
    L.push(`    return Xtr, ytr.astype("int64"), Xte, yte.astype("int64")`);
  } else {   // MNIST / FashionMNIST / KMNIST — idx-ubyte format
    L.push(`_BASE = {"MNIST": "https://ossci-datasets.s3.amazonaws.com/mnist/",`);
    L.push(`         "FashionMNIST": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",`);
    L.push(`         "KMNIST": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/"}`);
    L.push(`_FILES = {"tri": "train-images-idx3-ubyte.gz", "trl": "train-labels-idx1-ubyte.gz",`);
    L.push(`          "tei": "t10k-images-idx3-ubyte.gz",  "tel": "t10k-labels-idx1-ubyte.gz"}`);
    L.push(`def _idx(path):`);
    L.push(`    with gzip.open(path, "rb") as f: buf = f.read()`);
    L.push(`    magic = struct.unpack(">I", buf[:4])[0]; nd = magic & 0xFF`);
    L.push(`    dims = struct.unpack(">" + "I" * nd, buf[4:4 + 4 * nd])`);
    L.push(`    return np.frombuffer(buf[4 + 4 * nd:], dtype=np.uint8).reshape(dims)`);
    L.push(`def load_dataset(name, root="./data"):`);
    L.push(`    d = os.path.join(root, name); base = _BASE[name]`);
    L.push(`    p = {k: _fetch(base + v, os.path.join(d, v)) for k, v in _FILES.items()}`);
    L.push(`    Xtr = _idx(p["tri"]).astype("float32")[:, None] / 255.0   # (N, 1, 28, 28)`);
    L.push(`    Xte = _idx(p["tei"]).astype("float32")[:, None] / 255.0`);
    L.push(`    ytr = _idx(p["trl"]).astype("int64"); yte = _idx(p["tel"]).astype("int64")`);
    L.push(`    return Xtr, ytr, Xte, yte`);
  }
  return L;
}

function recordingConfig() {
  const aspects = [];
  if ($("#asp-weights").checked) aspects.push("weights");
  if ($("#asp-activations").checked) aspects.push("activations");
  let selection;               // python expr for layer_selection
  if ($("#record-mode").value === "preset") {
    selection = pyStr($("#record-preset").value);
  } else {
    const names = $$("#record-explicit input:checked").map(c => c.dataset.name);
    selection = names.length ? pyList(names) : "None";
  }
  return { aspects, selection, pooling: $("#activation-pooling").value };
}

function generate() {
  const warns = [];
  traceShapes();                       // ensure per-layer _in/_out are current
  const dsName = $("#dataset").value;
  const ds = DATASETS[dsName];
  const rec = recordingConfig();
  const device = $("#device").value;
  const epochs = $("#epochs").value || "10";
  // Trajectory snapshot cadence: per epoch (default) or per N mini-batches.
  const snapEvery = +($("#snapshot-every").value) || 1;
  const snapPerBatch = $("#snapshot-sched").value === "iteration";
  const snapArgs = `snapshot_every=${snapEvery}`
                 + (snapPerBatch ? `, snapshot_schedule="iteration"` : ``);
  const instances = Math.max(1, parseInt($("#instances").value) || 1);
  const lr = $("#lr").value || "1e-3";
  const optName = $("#optimizer").value;
  const batch = $("#batch").value || "128";
  const trainN = $("#train-n").value || "2000";
  const testN = $("#test-n").value || "500";

  // ---- analysis config ----
  let pipeExpr, traj = false, rep = null, repRaw = false, plotKind = null, needsLoss = false, presetKey = null;
  let overtrain = false, otFn = null, otArgs = null, otAspects = null, otPre = [];
  if (state.analysisMode === "overtraining") {
    overtrain = true; traj = true;
    const conf = OVERTRAIN[$("#overtrain-track").value];
    const v = collectOvertrain();
    otFn = conf.fn;
    let extraKw = "";
    if (conf.construction) {
      const cb = otConstruction(v);
      extraKw = cb.kw; otPre = cb.pre; otAspects = cb.aspects;
    } else {
      otAspects = (conf.aspectsFn ? conf.aspectsFn(v) : conf.aspects).slice();
    }
    otArgs = conf.call(v) + extraKw;
    if (conf.validate) conf.validate(v, epochs, warns);
    if (conf.construction && v.cons === "activation cloud")
      warns.push("activation-cloud construction: make sure the chosen layer is in the recorded set (activations are captured only for recorded layers). "
        + "With many neurons/samples, per-epoch H1 + Wasserstein can be very slow — prefer dim 0, wide layers as neurons-points sparingly, or fewer epochs first.");
  } else if (state.analysisMode === "preset") {
    const p = PRESETS.find(x => x.k === $("#preset").value);
    presetKey = p.k; traj = !!p.traj; rep = p.rep || null; needsLoss = !!p.loss;
    pipeExpr = `TDAPipeline.from_paper(${pyStr(p.k)})`;
    if (p.note) warns.push(`Preset “${p.k}”: ${p.note}`);
    // ── warn about obviously-incompatible configuration ──
    const need = PRESET_NEEDS[p.k] || {};
    const inst = +($("#instances").value || 1);
    if (need.act && !$("#asp-activations").checked)
      warns.push(`Preset “${p.k}” analyses activations — tick “activations” under Record what (it's currently off), otherwise extraction captures none and the run will fail.`);
    if (need.weights && !$("#asp-weights").checked)
      warns.push(`Preset “${p.k}” needs weights — tick “weights” under Record what.`);
    if (need.multi && inst < 2)
      warns.push(`Preset “${p.k}” pools across several runs/models — set Instances ≥ 2 (currently ${inst}); with one instance it takes the wrong code path and errors.`);
    if (need.conv3 && !state.layers.some(L => L.type === "conv2d" && +L.params.kernel === 3))
      warns.push(`Preset “${p.k}” harvests 3×3 conv kernels — your model has no 3×3 Conv2d layer, so none will be found.`);
    if (need.longTraj && !snapPerBatch)
      warns.push(`Preset “${p.k}” estimates a fractal dimension of the training trajectory, which assumes a long trajectory (the paper studies thousands of mini-batch iterates, not epochs). Set “Snapshot every” to per-batch (e.g. every 1–5 batches) for a stable estimate — per-epoch gives too few points and the fit may error (a valid dimension needs a positive slope).`);
  } else {
    const builder = $("#builder").value;
    const tool = $("#tool").value;
    const bkw = collectKwargs("bk");
    const tkw = collectKwargs("tk");
    plotKind = $("#plot-kind").value;
    const est = tkw.estimator ? tkw.estimator.replace(/"/g,"") : null;
    const method = tool === "dimension" ? (getKw("tk","method")||"") : null;
    if (tool === "dimension") {
      traj = est === "trajectory_dimension";
      tkw.method = pyStr(method);
      if (builder !== "None") warns.push("The dimension tool runs directly on the data — builder forced to None.");
    }
    // dimension tools bypass Module 1 (no graph builder)
    const effectiveBuilder = tool === "dimension" ? "None" : builder;
    const builderPy = effectiveBuilder === "None" ? "None" : pyStr(effectiveBuilder);
    const parts = [`builder=${builderPy}`];
    if (effectiveBuilder !== "None" && Object.keys(bkw).length) parts.push(`builder_kwargs=${dictLiteral(bkw)}`);
    parts.push(`tool=${pyStr(tool)}`);
    if (Object.keys(tkw).length) parts.push(`tool_kwargs=${dictLiteral(tkw)}`);
    pipeExpr = `TDAPipeline(${parts.join(", ")})`;
    // representation
    const repSel = $("#representation").value;
    if (repSel === "__kernels__") {
      const kl = $("#rep-layer").value || "conv1";
      rep = `lambda snap: snap.kernel_weight("${kl}")`;   // raw python: conv filters as points
      repRaw = true;
      if (!state.layers.some(l => l.type === "conv2d" && l.name === kl))
        warns.push(`representation "conv kernels": pick a Conv2d layer (got "${kl}").`);
    } else {
      rep = repSel === "__layer__" ? $("#rep-layer").value : repSel;
    }
    if (builder === "weight_graph" && ["correlation","weighted_activation"].includes((getKw("bk","edge_weight")||"")) && rep !== "coupled")
      warns.push('edge_weight "correlation"/"weighted_activation" needs representation = coupled (weight, activation).');
    if (builder === "weight_graph" && (getKw("bk","induced_paths")||"no") === "yes" && (getKw("bk","graph_scope")||"") !== "full")
      warns.push("induced_paths=True requires graph_scope=full.");
    if (builder === "labelled_complex_graph" && rep !== "inputs_labels")
      warns.push("labelled_complex_graph expects representation = inputs + predicted labels.");
    if (builder === "polyhedral_graph" && (rep === "weights" || rep === "coupled"))
      warns.push("polyhedral_graph reads activation patterns — use representation = activations or a specific layer.");
    if (builder === "kernel_graph" && !repRaw)
      warns.push('kernel_graph expects representation = "conv kernels of a layer…".');
  }

  // ---- aspects must satisfy representation (snapshot mode) ----
  let aspects = rec.aspects.slice();
  if (overtrain) aspects = otAspects.slice();
  else if (!traj && rep) {
    const need = repRaw ? ["weights"]
      : rep === "weights" ? ["weights"]
      : rep === "activations" ? ["activations"]
      : rep === "coupled" ? ["weights","activations"]
      : rep === "inputs_labels" ? ["activations","classifications"]
      : ["activations"];
    need.forEach(a => { if (!aspects.includes(a)) { aspects.push(a); } });
  }
  if (traj && aspects.length === 0) aspects = ["weights"];
  if (aspects.length === 0) aspects = ["weights"];

  if (state.layers.length === 0) warns.push("Add at least one layer to the model.");
  if (rec.selection === "None" && $("#record-mode").value === "explicit") warns.push("No layers selected to record.");
  if (traj && !overtrain && (+epochs) < 10) warns.push("Trajectory-dimension analyses need ≥10 snapshots — set Epochs to at least 10 (the method assumes a long trajectory; more is better).");
  if (overtrain && (+epochs) < 2) warns.push("Over-training plots need at least 2 epochs (one snapshot per epoch).");

  // Mapper sweep: a population plus a parameter grid, via MapperStudy.
  const sweep = state.analysisMode === "sweep";
  const sw = sweep ? collectSweep(warns) : null;
  if (sweep) {
    if (state.framework === "tf")
      warns.push("The Mapper sweep generator emits PyTorch only — switch the framework toggle to PyTorch.");
    if (sw.nConfigs > 2000)
      warns.push(`This grid expands to ${sw.nConfigs} configurations — expect a long run. `
               + `Untick an option, or run it and resume later.`);
    const cnt = $("#sweep-count");
    if (cnt) cnt.textContent = sw.nConfigs
      ? `Grid: ${sw.layers.length} layer × ${sw.views.length} view`
        + `${sw.untrained ? " × 2 (trained + control)" : ""} = ${sw.nClouds} cloud(s), `
        + `then × ${sw.prepCount} preprocessing × ${sw.filtCount} filter × ${sw.lenses.length} lens `
        + `× ${sw.intervals.length} resolution × ${sw.overlaps.length} overlap × ${sw.clusCount} clusterer `
        + `→ ${sw.nConfigs} configurations, from ${instances} trained network(s).`
      : "Nothing to run — tick at least one option in each group.";
  }

  // multi-model: train N instances and pool them
  const multi = instances > 1 && !traj && !overtrain && !sweep;       // snapshot pooling
  const gabKernels = multi && presetKey === "gabrielsson2019";
  // gabella2021: pool the *weight trajectories* of N runs from one shared init
  const gabellaMulti = instances > 1 && !overtrain && presetKey === "gabella2021" && state.framework !== "tf";
  if (instances > 1 && (traj || overtrain) && !gabellaMulti)
    warns.push("‘Model instances’ only applies to single-snapshot analyses (and the gabella2021 trajectory) — using one model.");
  if (instances > 1 && presetKey === "gabella2021" && state.framework === "tf")
    warns.push("Multiple training-trajectory runs (gabella2021) are generated for PyTorch only — using one model.");
  if (gabKernels && !state.layers.some(l => l.type === "conv2d"))
    warns.push("Conv-kernel Mapper (gabrielsson2019) needs Conv2d layers with 3×3 kernels in the model.");

  const interactive3d = analysisNeedsMapper() && $("#interactive3d").checked;
  if (state.analysisMode === "custom" && $("#builder").value === "activation_graph" && $("#tool").value === "ph"
      && !("max_neurons" in collectKwargs("bk")))
    warns.push("activation_graph + PH over many neurons (e.g. conv layers) can be very slow — set max_neurons (e.g. 200) to subsample.");
  if (state.analysisMode === "custom" && $("#builder").value === "point_cloud_graph")
    warns.push("point_cloud_graph reads a single weight matrix — record ONE layer (a specific layer, not a group like all_linear).");

  const tf = state.framework === "tf";
  let insertedFlatten = false;
  if (tf && hasSeq()) warns.push("Transformer layers have no single-layer Keras equivalent and are skipped in TensorFlow — use PyTorch for those.");
  if (tf && aspects.includes("activations"))
    warns.push("TensorFlow activation capture can be limited for some Sequential models (toolkit limitation) — weight-based analyses are the most reliable in TF.");

  // =============== build code ===============
  const L = [];
  L.push(`"""Generated by the TANC Visual Builder.`);
  L.push(`Run:  python this_script.py   (needs: ${tf ? "tensorflow" : "torch"}, tanc)`);
  L.push(`"""`);
  if (ds) L.push(`import os, gzip, struct, pickle, tarfile, urllib.request`);
  L.push(`import numpy as np`);
  if (tf) {
    L.push(`import tensorflow as tf`);
    L.push(`from tensorflow import keras`);
    L.push(`from tensorflow.keras import layers`);
  } else {
    L.push(`import torch`);
    L.push(`import torch.nn as nn`);
    L.push(`from collections import OrderedDict`);
  }
  L.push(`import matplotlib`);
  L.push(`matplotlib.use("Agg")            # headless; drop this line for an interactive window`);
  if (!overtrain && !sweep) L.push(`from tanc import TDAPipeline`);
  if (traj) L.push(`from tanc.model_extractor import TrainingExtractor`);
  if (overtrain) L.push(`from tanc.visualisation import ${otFn}`);
  if (sweep) {
    L.push(`from tanc.pipeline import MapperStudy`);
    L.push(`from tanc.topo_tools import (DBSCANCells, SingleLinkageCells,\n                                   FirstGapCells, HDBSCANCells, WardCells)`);
    L.push(`from tanc.visualisation import (plot_stability_heatmap,`);
    L.push(`                                      plot_cover_degeneracy,`);
    L.push(`                                      plot_node_size_distribution)`);
  }
  L.push(``);
  if (tf) L.push(`np.random.seed(0); tf.random.set_seed(0)`);
  else {
    L.push(`torch.manual_seed(0); np.random.seed(0)`);
    if (device === "auto")
      L.push(`DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")`);
    else L.push(`DEVICE = ${pyStr(device)}`);
  }
  L.push(``);

  // SoRO class
  if (hasSoRO() && !tf) {
    L.push(`# ── SoRO (Sum-of-Rank-One) factored FC layer: W = U·diag(sigma)·Vᵀ ──`);
    L.push(`class SoRO(nn.Module):`);
    L.push(`    def __init__(self, in_features, out_features, rank):`);
    L.push(`        super().__init__()`);
    L.push(`        # U, V uniform random, scaled so columns have ~unit norm; sigma ~ Uniform(0, 1]`);
    L.push(`        self.U = nn.Parameter((torch.rand(out_features, rank) * 2 - 1) * (3.0 / out_features) ** 0.5)  # (out, r)`);
    L.push(`        self.sigma = nn.Parameter(1.0 - torch.rand(rank))                                              # (r,)`);
    L.push(`        self.V = nn.Parameter((torch.rand(in_features, rank) * 2 - 1) * (3.0 / in_features) ** 0.5)    # (in, r)`);
    L.push(`    def forward(self, x):`);
    L.push(`        return ((x @ self.V) * self.sigma) @ self.U.t()`);
    L.push(`    def soro_factors(self):        # protocol the toolkit recognises`);
    L.push(`        return self.U, self.sigma, self.V`);
    L.push(``);
  }
  if (hasSoRO() && tf) {
    L.push(`# ── SoRO (Sum-of-Rank-One) factored Dense layer for Keras ──`);
    L.push(`class SoRODense(keras.layers.Layer):`);
    L.push(`    def __init__(self, units, rank, **kw):`);
    L.push(`        super().__init__(**kw); self.units, self.rank = units, rank`);
    L.push(`    def build(self, input_shape):`);
    L.push(`        f = int(input_shape[-1])`);
    L.push(`        # U, V uniform random, scaled so columns have ~unit norm; sigma ~ Uniform(0, 1]`);
    L.push(`        lim_u, lim_v = (3.0 / self.units) ** 0.5, (3.0 / f) ** 0.5`);
    L.push(`        self.U = self.add_weight(shape=(self.units, self.rank), initializer=keras.initializers.RandomUniform(-lim_u, lim_u), name="U")`);
    L.push(`        self.sigma = self.add_weight(shape=(self.rank,), initializer=keras.initializers.RandomUniform(1e-6, 1.0), name="sigma")`);
    L.push(`        self.V = self.add_weight(shape=(f, self.rank), initializer=keras.initializers.RandomUniform(-lim_v, lim_v), name="V")`);
    L.push(`    def call(self, x):`);
    L.push(`        return tf.matmul(tf.matmul(x, self.V) * self.sigma, self.U, transpose_b=True)`);
    L.push(`    def soro_factors(self):`);
    L.push(`        return self.U, self.sigma, self.V`);
    L.push(``);
  }

  // model — build the inner layer lines once, then wrap as `model = …` or a factory
  L.push(`# ── Model ─────────────────────────────────────────────────────────`);
  let _flatN = 0;
  const items = [];
  state.layers.forEach(Ly => {
    const def = LAYERS[Ly.type];
    if (tf && !def.genTF) { warns.push(`${def.label} (${Ly.name}) is not supported by the TensorFlow generator — skipped.`); return; }
    if ((Ly.type === "linear" || Ly.type === "soro") && Ly._in && Ly._in.kind === "image") {
      items.push(tf ? `layers.Flatten(name=${pyStr("flatten" + (_flatN || ""))}),`
                    : `(${pyStr("flatten" + (_flatN || ""))}, nn.Flatten()),`);
      _flatN++; insertedFlatten = true;
    }
    items.push(tf ? `${def.genTF(Ly.params, pyStr(Ly.name))},`
                  : `(${pyStr(Ly.name)}, ${def.gen(Ly.params)}),`);
  });
  const ctorOpen = tf ? `keras.Sequential([` : `nn.Sequential(OrderedDict([`;
  const ctorClose = tf ? `])` : `]))`;
  const inputLine = (tf && ds) ? `keras.Input(shape=(${ds.hw}, ${ds.hw}, ${ds.ch})),   # channels-last` : null;
  if (multi || gabellaMulti || sweep) {
    L.push(`def build_model():                 # a fresh model per instance`);
    L.push(`    return ${ctorOpen}`);
    if (inputLine) L.push(`        ${inputLine}`);
    items.forEach(it => L.push(`        ${it}`));
    L.push(`    ${ctorClose}`);
  } else {
    L.push(`model = ${ctorOpen}`);
    if (inputLine) L.push(`    ${inputLine}`);
    items.forEach(it => L.push(`    ${it}`));
    L.push(ctorClose);
  }
  if (insertedFlatten) warns.push("Auto-inserted a Flatten layer before Linear/SoRO layers that receive image input.");
  L.push(``);

  // data
  L.push(`# ── Data (downloaded & parsed with numpy — no torchvision needed) ──`);
  if (ds) {
    datasetLoaderLines(dsName).forEach(x => L.push(x));
    L.push(`_Xtr, _ytr, _Xte, _yte = load_dataset(${pyStr(dsName)})`);
    L.push(`_Xtr, _ytr = _Xtr[:${trainN}], _ytr[:${trainN}]`);
    L.push(`_Xte, _yte = _Xte[:${testN}], _yte[:${testN}]`);
    if (tf) {
      L.push(`X_tr = np.transpose(_Xtr, (0, 2, 3, 1)); y_tr = _ytr   # NCHW → NHWC for Keras`);
      L.push(`X_te = np.transpose(_Xte, (0, 2, 3, 1)); y_te = _yte`);
    } else {
      L.push(`X_tr = torch.tensor(_Xtr); y_tr = torch.tensor(_ytr)`);
      L.push(`X_te = torch.tensor(_Xte); y_te = torch.tensor(_yte)`);
    }
  } else {
    const src = $("#custom-data-src").value || "YOUR_DATASET";
    L.push(`# TODO: load your dataset (${src}) into X_tr/y_tr/X_te/y_te`);
    L.push(`#   ${tf ? "X_* : numpy (N, H, W, C) float" : "X_* : (N, C, H, W) float tensor"};  y_* : (N,) int`);
    L.push(`raise NotImplementedError("Fill in the loader for: ${src}")`);
    warns.push("Custom dataset: a loader stub was emitted — fill in X_tr/y_tr/X_te/y_te.");
  }
  L.push(``);

  const optTorch = `torch.optim.${optName}(model.parameters(), lr=${lr})`;
  const optKeras = `keras.optimizers.${optName}(learning_rate=${lr})`;
  const compileKw = `{"optimizer": ${optKeras}, "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]}`;

  if (sweep) {
    // ---- MapperStudy: a seed population, then Mapper across a parameter grid ----
    L.push(`# ── Mapper sweep ──────────────────────────────────────────────────`);
    L.push(`# Trains ${instances} network(s), reads each layer as a point cloud, and runs`);
    L.push(`# Mapper across the grid below. Every graph is reported alongside the nerve`);
    L.push(`# of its own cover, so b1_excess = b1 - nerve_b1 separates a real finding`);
    L.push(`# from a property of the cover. Overlap is |I ∩ I'| / |I| (KeplerMapper /`);
    L.push(`# giotto-tda convention).`);
    L.push(``);
    L.push(`study = MapperStudy(`);
    L.push(`    model_fn     = build_model,`);
    L.push(`    train_data   = (X_tr, y_tr),`);
    L.push(`    val_data     = (X_te, y_te),`);
    L.push(`    extract_data = X_te[:64],`);
    L.push(`    n_models     = ${instances},`);
    L.push(`    epochs       = ${epochs},`);
    L.push(`    batch_size   = ${batch},`);
    L.push(`    criterion    = nn.CrossEntropyLoss(),`);
    L.push(`    optimizer_fn = lambda m: torch.optim.${optName}(m.parameters(), lr=${lr}),`);
    L.push(`    device       = DEVICE,`);
    if (sw.untrained) L.push(`    include_untrained = True,   # matched control at 0 epochs`);
    L.push(``);
    L.push(`    layer        = ${pyAxis(sw.layers, true)},`);
    L.push(`    view         = ${pyAxis(sw.views, true)},`);
    L.push(`    preprocess   = ${sw.prepExpr},        # rescaling, before the lens`);
    L.push(`    point_filter = ${sw.filtExpr},        # point removal, after the lens (cover pinned)`);
    L.push(`    lens         = ${pyAxis(sw.lenses, true)},`);
    L.push(`    n_intervals  = ${pyAxis(sw.intervals, false)},`);
    L.push(`    overlap      = ${pyAxis(sw.overlaps, false)},`);
    L.push(`    clusterer    = ${sw.clusExpr},`);
    L.push(`)`);
    L.push(``);
    L.push(`# Checked before a single model is built: that training will capture what`);
    L.push(`# the grid asks for, and that no configuration is self-contradictory.`);
    L.push(`study.validate()`);
    L.push(``);
    L.push(`result = study.run("runs/sweep")     # never overwrites; prints the path used`);
    L.push(``);
    L.push(`# ── What came out ─────────────────────────────────────────────────`);
    L.push(`rows = result.rows()`);
    L.push(`print(f"\\n{len(rows)} configurations ran, "`);
    L.push(`      f"{len(result.rejected)} rejected, {len(result.errors)} errored")`);
    L.push(``);
    L.push(`# Configurations whose topology is indistinguishable. A structure that`);
    L.push(`# survives a RANGE of parameters is evidence; one that appears at a single`);
    L.push(`# setting is a parameter accident.`);
    L.push(`for grp in result.plateaus(min_size=2)[:3]:`);
    L.push(`    print(f"  {grp['n_configs']:>3} configs share {grp['signature']}")`);
    L.push(`    for axis, vals in grp["spans"].items():`);
    L.push(`        print(f"        {axis}: {', '.join(vals)}")`);
    L.push(``);
    L.push(`# Best-scoring graphs that are also legible. The legibility filter runs`);
    L.push(`# FIRST: ranking on b1_excess alone selects for shattered graphs, whose`);
    L.push(`# huge b1 is just E - V + b0 over a dust of singleton nodes.`);
    L.push(`print("\\nleading configurations:")`);
    L.push(`for r in result.leading(measure="b1_excess", min_node_median=5, n=5):`);
    L.push(`    print(f"  {r['cloud']:<20} res={r['n_intervals']:<4} ovl={r['overlap']:<5} "`);
    L.push(`          f"b1={r['b1']:<6} nerve={r['nerve_b1']:<6} excess={r['b1_excess']}")`);
    L.push(``);
    L.push(`# ── Diagnostics ───────────────────────────────────────────────────`);
    L.push(`plot_stability_heatmap(rows, measure="b1_excess").savefig("tda_result_0.png", dpi=130)`);
    L.push(`plot_cover_degeneracy(rows).savefig("tda_result_1.png", dpi=130)`);
    L.push(`plot_node_size_distribution(rows).savefig("tda_result_2.png", dpi=130)`);
    L.push(`print("\\nsaved figures -> tda_result_0.png (stability), "`);
    L.push(`      "tda_result_1.png (cover degeneracy), tda_result_2.png (node sizes)")`);
    return { code: L.join("\n"), warns };
  }

  if (gabellaMulti) {
    // ---- gabella2021: pool the weight trajectories of N runs from one init ----
    L.push(`# ── Train ${instances} runs from one shared init → pool weight trajectories ──`);
    L.push(`import copy`);
    L.push(`base = build_model()               # shared initialisation (common root)`);
    L.push(`def _run_trajectory(seed):`);
    L.push(`    model = copy.deepcopy(base).to(DEVICE)`);
    L.push(`    torch.manual_seed(seed)`);
    L.push(`    ext = TrainingExtractor(`);
    L.push(`        model=model, train_data=(X_tr, y_tr), val_data=(X_te, y_te), batch_size=${batch},`);
    L.push(`        criterion=nn.CrossEntropyLoss(), optimizer=${optTorch},`);
    L.push(`        extract_data=X_te[:64], aspects=["weights"], ${snapArgs}, device=DEVICE, clarify=False)`);
    L.push(`    view = ext.run(epochs=${epochs}, target_accuracy=None, verbose=False)`);
    L.push(`    return np.stack([np.concatenate([w.ravel() for w in snap_w]) for snap_w in view.weight_trajectory()])`);
    L.push(`trajectories = [_run_trajectory(s) for s in range(${instances})]`);
    L.push(`data = np.concatenate(trajectories, axis=0)`);
    L.push(`run_id = np.concatenate([np.full(len(t), i) for i, t in enumerate(trajectories)])`);
    L.push(`print(f"pooled {data.shape[0]} trajectory points from ${instances} runs")`);
    L.push(``);
    L.push(`pipe = ${pipeExpr}`);
    L.push(`result = pipe.fit(data)`);
    L.push(``);
    L.push(`# ── Mapper of the pooled trajectories, coloured by run ────────────`);
    if (interactive3d) {
      mapperHtmlHelperLines().forEach(x => L.push(x));
      L.push(`_mapper_html(result, "tda_result_0.html", color_by=run_id)`);
      L.push(`print("saved interactive graph -> tda_result_0.html")`);
    } else {
      L.push(`fig = result.plot("graph", color_by=run_id, save="tda_result_0.png")`);
      L.push(`print("saved figure -> tda_result_0.png")`);
    }
    return { code: L.join("\n"), warns };
  }

  if (multi) {
    // ---- Train N instances (different seeds) and pool them ----
    L.push(`# ── Train ${instances} model instances (different seeds) ───────────`);
    if (!tf) {
      L.push(`from torch.utils.data import DataLoader, TensorDataset`);
      L.push(`_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=${batch}, shuffle=True)`);
      L.push(`_crit = nn.CrossEntropyLoss()`);
      L.push(`@torch.no_grad()`);
      L.push(`def _acc(model, X, y, bs=512):`);
      L.push(`    correct = sum((model(X[i:i+bs].to(DEVICE)).argmax(1).cpu() == y[i:i+bs]).sum().item()`);
      L.push(`                  for i in range(0, len(X), bs))`);
      L.push(`    return correct / len(X)`);
      L.push(`models = []`);
      L.push(`for seed in range(${instances}):`);
      L.push(`    torch.manual_seed(seed)`);
      L.push(`    model = build_model().to(DEVICE)`);
      L.push(`    _opt = ${optTorch}`);
      L.push(`    for epoch in range(${epochs}):`);
      L.push(`        model.train()`);
      L.push(`        for xb, yb in _loader:`);
      L.push(`            xb, yb = xb.to(DEVICE), yb.to(DEVICE)`);
      L.push(`            _opt.zero_grad(); _crit(model(xb), yb).backward(); _opt.step()`);
      L.push(`        print(f"__EPOCH__ {seed * ${+epochs} + epoch + 1}/${(+instances) * (+epochs)}", flush=True)`);
      L.push(`    model.eval(); models.append(model)`);
      L.push(`    print(f"instance {seed+1}/${instances}  train {_acc(model, X_tr, y_tr):.3f}  test {_acc(model, X_te, y_te):.3f}")`);
    } else {
      L.push(`models = []`);
      L.push(`for seed in range(${instances}):`);
      L.push(`    tf.random.set_seed(seed)`);
      L.push(`    model = build_model()`);
      L.push(`    model.compile(optimizer=${optKeras}, loss="sparse_categorical_crossentropy", metrics=["accuracy"])`);
      L.push(`    _cb = keras.callbacks.LambdaCallback(on_epoch_end=lambda e, _l: print(f"__EPOCH__ {seed * ${+epochs} + e + 1}/${(+instances) * (+epochs)}", flush=True))`);
      L.push(`    model.fit(X_tr, y_tr, epochs=${epochs}, batch_size=${batch}, verbose=0, callbacks=[_cb])`);
      L.push(`    models.append(model)`);
      L.push(`    _te = model.evaluate(X_te, y_te, verbose=0)`);
      L.push(`    print(f"instance {seed+1}/${instances}  test {_te[1]:.3f}")`);
    }
    L.push(``);
    L.push(`pipe = ${pipeExpr}`);
    if (gabKernels) {
      L.push(`# ── Pool 3×3 conv kernels from all models → Mapper ────────────────`);
      L.push(`from tanc.model_extractor import extract_model`);
      L.push(`def harvest_kernels(model):`);
      L.push(`    snap = extract_model(model, X_te[:64], aspects=["weights"], layer_selection="all_conv", clarify=False)`);
      L.push(`    mats = [np.asarray(w, float).reshape(-1, 9) for w in snap.kernel_weights.values() if w.shape[-2:] == (3, 3)]`);
      L.push(`    return np.concatenate(mats, axis=0) if mats else np.empty((0, 9))`);
      L.push(`K = np.concatenate([harvest_kernels(m) for m in models], axis=0)`);
      L.push(`K = K - K.mean(1, keepdims=True)                            # remove DC term`);
      L.push(`K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)  # onto the unit sphere`);
      L.push(`print(f"pooled {len(K)} conv kernels from ${instances} models")`);
      L.push(`result = pipe.fit(K)`);
    } else {
      L.push(`# ── Pool the representation across models, then analyse ───────────`);
      L.push(`result = pipe.fit_models(models, X_te, representation=${pyStr(rep || "weights")}, layer_idx=0)`);
    }
    L.push(``);
    resultsLines(interactive3d, plotKind).forEach(x => L.push(x));
    return { code: L.join("\n"), warns };
  }

  if (!traj) {
    // ---- Template A: train, then extract + analyse a snapshot ----
    L.push(`# ── Train ─────────────────────────────────────────────────────────`);
    if (!tf) {
      L.push(`from torch.utils.data import DataLoader, TensorDataset`);
      L.push(`model = model.to(DEVICE)`);
      L.push(`_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=${batch}, shuffle=True)`);
      L.push(`_opt = ${optTorch}`);
      L.push(`_crit = nn.CrossEntropyLoss()`);
      L.push(`for epoch in range(${epochs}):`);
      L.push(`    model.train()`);
      L.push(`    for xb, yb in _loader:`);
      L.push(`        xb, yb = xb.to(DEVICE), yb.to(DEVICE)`);
      L.push(`        _opt.zero_grad(); loss = _crit(model(xb), yb); loss.backward(); _opt.step()`);
      L.push(`    print(f"epoch {epoch+1}/${epochs}  loss={loss.item():.4f}")`);
      L.push(`model.eval()`);
      L.push(`@torch.no_grad()`);
      L.push(`def _accuracy(X, y, bs=512):`);
      L.push(`    correct = 0`);
      L.push(`    for i in range(0, len(X), bs):`);
      L.push(`        pred = model(X[i:i+bs].to(DEVICE)).argmax(1).cpu()`);
      L.push(`        correct += (pred == y[i:i+bs]).sum().item()`);
      L.push(`    return correct / len(X)`);
      L.push(`print(f"final accuracy — train: {_accuracy(X_tr, y_tr):.4f}   test: {_accuracy(X_te, y_te):.4f}")`);
    } else {
      L.push(`model.compile(optimizer=${optKeras}, loss="sparse_categorical_crossentropy", metrics=["accuracy"])`);
      L.push(`model.fit(X_tr, y_tr, epochs=${epochs}, batch_size=${batch}, validation_data=(X_te, y_te), verbose=2)`);
      L.push(`_tr = model.evaluate(X_tr, y_tr, verbose=0); _te = model.evaluate(X_te, y_te, verbose=0)`);
      L.push(`print(f"final accuracy — train: {_tr[1]:.4f}   test: {_te[1]:.4f}")`);
    }
    L.push(``);
    L.push(`# ── Topological analysis ──────────────────────────────────────────`);
    L.push(`pipe = ${pipeExpr}`);
    const fmArgs = [`model`, `X_te`, `aspects=${pyList(aspects)}`, `layer_selection=${rec.selection}`];
    if (rep) fmArgs.push(`representation=${repRaw ? rep : pyStr(rep)}`);
    if (hasSeq() && !tf && aspects.includes("activations")) fmArgs.push(`activation_pooling=${pyStr(rec.pooling)}`);
    L.push(`result = pipe.fit_model(`);
    L.push(`    ${fmArgs.join(",\n    ")},`);
    L.push(`)`);
  } else {
    // ---- Template B: train + capture trajectory, then analyse the view ----
    L.push(`# ── Train while capturing a per-epoch trajectory ──────────────────`);
    L.push(`extractor = TrainingExtractor(`);
    L.push(`    model=model,`);
    L.push(`    train_data=(X_tr, y_tr), val_data=(X_te, y_te), batch_size=${batch},`);
    if (!tf) {
      L.push(`    criterion=nn.CrossEntropyLoss(),`);
      L.push(`    optimizer=${optTorch},`);
    } else {
      L.push(`    compile_kwargs=${compileKw},`);
    }
    if (needsLoss && !tf) L.push(`    loss_eval_data=(X_te, y_te),   # per-sample losses for the loss-PH metric`);
    L.push(`    extract_data=X_te,`);
    L.push(`    aspects=${pyList(aspects)}, layer_selection=${rec.selection},`);
    L.push(`    ${snapArgs}, ${tf ? "" : "device=DEVICE, "}clarify=False,`);
    L.push(`)`);
    L.push(`view = extractor.run(epochs=${epochs}, target_accuracy=None, verbose=True)`);
    L.push(`_tr = view.train_accuracy_trajectory(); _va = view.accuracies()`);
    L.push(`print(f"final accuracy — train: {_tr[-1]:.4f}   test: {_va[-1]:.4f}")`);
    L.push(``);
    if (overtrain) {
      // per-epoch PH → trajectory plot (a visualisation function, not a pipeline)
      L.push(`# ── Persistent homology per epoch → trajectory plot ───────────────`);
      otPre.forEach(x => L.push(x));
      L.push(`fig = ${otFn}(${otArgs})`);
      L.push(`fig.savefig("tda_result_0.png", bbox_inches="tight")`);
      L.push(`print("saved figure -> tda_result_0.png")`);
      return { code: L.join("\n"), warns };
    }
    L.push(`# ── Topological analysis over the trajectory ──────────────────────`);
    L.push(`pipe = ${pipeExpr}`);
    L.push(`result = pipe.fit(view)`);
  }
  L.push(``);

  // results + plot (static PNG, or interactive 3-D Mapper HTML)
  resultsLines(interactive3d, plotKind).forEach(x => L.push(x));

  return { code: L.join("\n"), warns };
}

// ─────────────────────────────────────────────────────────────
// Syntax highlight (lightweight) + output
// ─────────────────────────────────────────────────────────────
function highlight(code) {
  const esc = code.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  return esc.split("\n").map(line => {
    let m;
    if ((m = line.match(/^(\s*)(#.*)$/))) return `${m[1]}<span class="tok-com">${m[2]}</span>`;
    // strings
    line = line.replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, '<span class="tok-str">$1</span>');
    // keywords
    line = line.replace(/\b(import|from|for|in|if|else|elif|return|class|def|None|True|False|as|raise|with)\b/g, '<span class="tok-kw">$1</span>');
    // numbers
    line = line.replace(/\b(\d+\.?\d*(e-?\d+)?)\b/g, '<span class="tok-num">$1</span>');
    return line;
  }).join("\n");
}

let _code = "";
function regen() {
  updateShapeHint(); updatePoolingVisibility();
  const { code, warns } = generate();
  _code = code;
  $("#code-out").innerHTML = highlight(code);
  const wbox = $("#warnings"); wbox.innerHTML = "";
  warns.forEach(w => { const d = el("div", "warn", w); wbox.appendChild(d); });
  $("#interactive-wrap").classList.toggle("hidden", !analysisNeedsMapper());   // 3-D toggle: Mapper only
  updateKernelNote();          // Mapper/gtda hint depends on the chosen analysis
}

// ─────────────────────────────────────────────────────────────
// Wiring
// ─────────────────────────────────────────────────────────────
function wire() {
  renderPalette(); renderPresets(); renderToolKwargs(); renderBuilderKwargs(); renderOvertrainTrack(); renderAll();

  $("#dataset").onchange = () => { $("#custom-data").classList.toggle("hidden", $("#dataset").value !== "custom"); renderAll(); regen(); updateKernelNote(); };
  ["#batch","#train-n","#test-n","#epochs","#instances","#lr","#optimizer","#device","#custom-data-src","#record-preset","#activation-pooling","#snapshot-every","#snapshot-sched"]
    .forEach(s => { const e = $(s); e.oninput = regen; e.onchange = regen; });

  $("#record-mode").onchange = () => {
    const explicit = $("#record-mode").value === "explicit";
    $("#record-explicit").classList.toggle("hidden", !explicit);
    $("#record-preset-wrap").classList.toggle("hidden", explicit);
    regen();
  };
  $("#asp-weights").onchange = regen;
  $("#asp-activations").onchange = () => { updatePoolingVisibility(); regen(); };

  // framework toggle
  $$("#framework button").forEach(b => b.onclick = () => {
    state.framework = b.dataset.fw;
    $$("#framework button").forEach(x => x.classList.toggle("active", x === b));
    updateKernelNote(); regen();
  });

  // analysis mode toggle
  $$("#analysis-mode button").forEach(b => b.onclick = () => {
    state.analysisMode = b.dataset.mode;
    $$("#analysis-mode button").forEach(x => x.classList.toggle("active", x === b));
    $("#analysis-preset").classList.toggle("hidden", state.analysisMode !== "preset");
    $("#analysis-custom").classList.toggle("hidden", state.analysisMode !== "custom");
    $("#analysis-overtraining").classList.toggle("hidden", state.analysisMode !== "overtraining");
    $("#analysis-sweep")?.classList.toggle("hidden", state.analysisMode !== "sweep");
    regen();
  });
  $$("#analysis-sweep input, #analysis-sweep select").forEach(e => {
    e.oninput = e.onchange = () => {
      if (e.id === "sweep-filter")    renderFilterStrength();
      if (e.id === "sweep-clusterer") renderQuantileVisibility();
      regen();
    };
  });
  renderFilterStrength();
  renderQuantileVisibility();
  const fs = $("#sweep-filter-strength");
  if (fs) fs.addEventListener("input", () => { fs.dataset.touched = "1"; });
  $("#overtrain-track").onchange = () => { renderOvertrainFields(); regen(); };
  $("#interactive3d").onchange = regen;
  $("#preset").onchange = () => { updatePresetDesc(); regen(); };
  $("#builder").onchange = () => { renderBuilderKwargs(); regen(); };
  $("#tool").onchange = () => { renderToolKwargs(); regen(); };
  $("#representation").onchange = () => { onRepChange(); regen(); };
  $("#plot-kind").onchange = regen;
  $("#rep-layer").onchange = regen;

  // code actions
  $("#copy-code").onclick = async () => {
    try { await navigator.clipboard.writeText(_code); flash("#copy-code", "Copied!"); }
    catch { flash("#copy-code", "Press ⌘/Ctrl+C"); }
  };
  $("#download-code").onclick = () => {
    const blob = new Blob([_code], {type:"text/x-python"});
    const a = el("a"); a.href = URL.createObjectURL(blob); a.download = "tda_experiment.py"; a.click();
    URL.revokeObjectURL(a.href);
  };
  $("#run-code").onclick = runCode;
  $("#close-run").onclick = () => $("#run-output").classList.add("hidden");
  $("#refresh-kernels").onclick = loadKernels;

  // theme
  $("#theme-toggle").onclick = () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "dark" ? "light" : cur === "light" ? "dark" : "dark";
    document.documentElement.setAttribute("data-theme", next);
  };
  $("#reset-all").onclick = () => { state.layers = []; renderAll(); regen(); };
  $("#load-example").onclick = loadExample;

  loadExample();   // start with something meaningful
  loadKernels();   // detect Python environments (if the runner backend is up)
}

function flash(sel, txt) {
  const b = $(sel); const old = b.textContent; b.textContent = txt;
  setTimeout(() => b.textContent = old, 1200);
}

function loadExample() {
  // The LeNet of Sandler's SoRO study (WL's standard LeNet on MNIST):
  // Conv 20@5x5 -> ReLU -> MaxPool 2 -> Conv 50@5x5 -> ReLU -> MaxPool 2 ->
  // FC-1 800x500 -> ReLU -> FC-2 500x10 (Flatten auto-inserted before fc1).
  // touched:"padding" pins padding=0 (otherwise the same-conv autodef overrides it)
  const L = (type, params, name, touched=[]) => ({ id:_uid++, type, params, name, touched:new Set(touched) });
  state.layers = [
    L("conv2d",    {in:1,  out:20, kernel:5, stride:1, padding:0}, "conv1", ["padding"]),
    L("relu",      {}, "relu1"),
    L("maxpool2d", {kernel:2}, "pool1"),
    L("conv2d",    {in:20, out:50, kernel:5, stride:1, padding:0}, "conv2", ["padding"]),
    L("relu",      {}, "relu2"),
    L("maxpool2d", {kernel:2}, "pool2"),
    L("linear",    {in:800, out:500}, "fc1"),
    L("relu",      {}, "relu3"),
    L("linear",    {in:500, out:10},  "fc2"),
  ];
  $("#dataset").value = "MNIST";
  $("#epochs").value = "20";                  // the paper's figure protocol
  // Default analysis: the study's headline — per-epoch diagram-distance churn
  state.analysisMode = "overtraining";
  $$("#analysis-mode button").forEach(x => x.classList.toggle("active", x.dataset.mode === "overtraining"));
  $("#analysis-preset").classList.add("hidden"); $("#analysis-custom").classList.add("hidden");
  $("#analysis-overtraining")?.classList.remove("hidden");
  $("#record-preset").value = "all_linear";
  $("#asp-weights").checked = true; $("#asp-activations").checked = false;
  updatePresetDesc(); renderOvertrainTrack(); renderAll(); regen();
}

// ─────────────────────────────────────────────────────────────
// Optional runner backend
// ─────────────────────────────────────────────────────────────
async function loadKernels() {
  const sel = $("#kernel"), note = $("#kernel-note");
  try {
    const resp = await fetch("/kernels");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const { kernels } = await resp.json();
    sel.innerHTML = ""; sel.disabled = false;
    if (!kernels.length) { sel.innerHTML = "<option>no interpreters found</option>"; sel.disabled = true; return; }
    const PKG = { torch:"torch", tensorflow:"tensorflow", tanc:"tanc", gtda:"giotto-tda" };
    kernels.forEach(k => {
      const have = ["torch","tensorflow","tanc","gtda"].filter(p => k[p]).map(p => PKG[p]);
      const o = el("option");
      o.value = k.path;
      o.dataset.torch = k.torch ? "1" : "0";
      o.dataset.tf = k.tensorflow ? "1" : "0";
      o.dataset.tda = k.tanc ? "1" : "0";
      o.dataset.gtda = k.gtda ? "1" : "0";
      o._label = `${k.label}${k.current ? " (current)" : ""}  [${have.join(", ") || "no relevant packages"}]`;
      o.title = k.path;
      sel.appendChild(o);
    });
    labelKernelOptions();
    // default: prefer a kernel usable for the framework AND with giotto-tda (Mapper)
    const opts = [...sel.options];
    const best = opts.find(o => kernelOk(o) && o.dataset.gtda === "1")
              || opts.find(o => kernelOk(o)) || opts[0];
    sel.value = best.value;
    updateKernelNote();
    sel.onchange = updateKernelNote;
  } catch {
    sel.innerHTML = "<option>start server (python web/server.py) to select</option>";
    sel.disabled = true; note.textContent = "";
  }
}
function kernelOk(o) {
  const fw = state.framework === "tf" ? o.dataset.tf : o.dataset.torch;
  return fw === "1" && o.dataset.tda === "1";
}
function labelKernelOptions() {
  [...$("#kernel").options].forEach(o => { if (o._label) o.textContent = (kernelOk(o) ? "✓ " : "•  ") + o._label; });
}
function analysisNeedsMapper() {
  if (state.analysisMode === "custom") return $("#tool").value === "mapper";
  return ["rathore2021","zhou2023","gabrielsson2019","gabella2021"].includes($("#preset").value);
}
function updateKernelNote() {
  const sel = $("#kernel");
  if (sel.disabled) { $("#kernel-note").textContent = ""; return; }   // no backend
  labelKernelOptions();
  const o = sel.selectedOptions[0];
  const fwName = state.framework === "tf" ? "tensorflow" : "torch";
  let msg = "";
  if (!(o && kernelOk(o))) msg = `⚠ selected env is missing ${fwName} / tanc`;
  else if (analysisNeedsMapper() && o.dataset.gtda !== "1")
    msg = "⚠ selected env has no giotto-tda — Mapper analyses will fail (pip install giotto-tda)";
  $("#kernel-note").textContent = msg;
}

async function runCode() {
  const out = $("#run-output"); out.classList.remove("hidden");
  const log = $("#run-log"); log.textContent = "Starting…";
  $("#run-figs").innerHTML = "";
  const bar = $("#run-progress"), fill = $("#run-progress-fill"), plab = $("#run-progress-label");
  bar.classList.add("hidden"); fill.style.width = "0%"; plab.textContent = "";
  const kernelSel = $("#kernel");
  const python = (!kernelSel.disabled && kernelSel.value) ? kernelSel.value : "";
  const epochsFallback = +($("#epochs").value || 0);   // total for markers that lack a /total

  // epoch-progress markers, most specific first:
  //   __EPOCH__ 3/10 (our codegen) · "Epoch 3/10" (keras) · "epoch 3/10 …" (torch loops)
  //   "[snapshot] epoch=3" (TrainingExtractor verbose — total from the Epochs field)
  const progress = t => {
    let cur = 0, tot = 0, m;
    if ((m = t.match(/__EPOCH__\s+(\d+)\s*\/\s*(\d+)/)))      { cur = +m[1]; tot = +m[2]; }
    else if ((m = t.match(/^\s*epoch\s+(\d+)\/(\d+)/i)))        { cur = +m[1]; tot = +m[2]; }
    else if ((m = t.match(/\[snapshot\]\s+epoch=(\d+)/)))        { cur = +m[1]; tot = epochsFallback; }
    if (!cur || !tot) return;
    bar.classList.remove("hidden");
    fill.style.width = Math.min(100, (100 * cur) / tot) + "%";
    plab.textContent = `epoch ${cur} / ${tot}`;
  };

  const lines = [];
  let meta = "", done = null;
  const render = () => { log.textContent = meta + lines.join("\n"); log.scrollTop = log.scrollHeight; };
  const handle = msg => {
    if (msg.type === "meta") { meta = `▶ ran with: ${msg.python}\n` + (msg.note || "") + "\n"; render(); }
    else if (msg.type === "line") { lines.push(msg.text); progress(msg.text); render(); }
    else if (msg.type === "done") done = msg;
  };

  try {
    const resp = await fetch("/run", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ code:_code, python }) });
    if (!resp.ok || !resp.body) throw new Error("HTTP " + resp.status);
    const reader = resp.body.getReader(); const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done: eof } = await reader.read();
      if (eof) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const raw = buf.slice(0, nl); buf = buf.slice(nl + 1);
        if (!raw.trim()) continue;
        try { handle(JSON.parse(raw)); } catch { handle({ type:"line", text: raw }); }
      }
    }
    if (!done && buf.trim()) {                      // older (non-streaming) server: one JSON blob
      try {
        const data = JSON.parse(buf);
        meta = data.python ? `▶ ran with: ${data.python}\n\n` : "";
        lines.push(data.stdout || "");
        if (data.stderr) lines.push("[stderr]\n" + data.stderr);
        done = { rc: 0, figures: data.figures || [], html: data.html || [] };
      } catch { /* leave as log */ }
    }
    if (done) {
      if (plab.textContent) { fill.style.width = "100%"; plab.textContent += "  ✓"; }
      if (done.timeout) lines.push("[timed out — raise TDA_RUN_TIMEOUT to allow longer runs]");
      else if (done.rc) lines.push(`[process exited with code ${done.rc}]`);
      render();
      (done.figures || []).forEach(b64 => { const img = el("img"); img.src = "data:image/png;base64," + b64; $("#run-figs").appendChild(img); });
      (done.html || []).forEach(h => {
        const iframe = el("iframe"); iframe.className = "run-iframe";
        iframe.src = URL.createObjectURL(new Blob([h], { type: "text/html" }));   // interactive 3-D graph
        $("#run-figs").appendChild(iframe);
      });
    }
  } catch (e) {
    log.textContent = "Could not reach the runner backend.\nStart it with:  python web/server.py\n\n(" + e.message + ")";
  }
}

document.addEventListener("DOMContentLoaded", wire);
