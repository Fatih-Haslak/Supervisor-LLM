"use strict";

const form = document.getElementById("task-form");
const runButton = document.getElementById("run-button");
const statusEl = document.getElementById("status");
const timeline = document.getElementById("timeline");
const toolsEl = document.getElementById("tools");
const agentsEl = document.getElementById("agents");
const reviewsEl = document.getElementById("reviews");
const approvalEl = document.getElementById("approval");
const approvalPreview = document.getElementById("approval-preview");
const errorEl = document.getElementById("ui-error");
const answerEl = document.getElementById("answer-text");
const flowGraph = document.getElementById("flow-graph");
const flowSummary = document.getElementById("flow-summary");
const flowInspector = document.getElementById("flow-inspector");
let stream = null;
let activeTask = null;
let selectedFlowNode = null;
let conversationId = localStorage.getItem("agent-conversation-id") || crypto.randomUUID();
let loadedTask = null;
localStorage.setItem("agent-conversation-id", conversationId);
const modeDescriptions = {
  auto: "Önerilen: model sohbeti tanır; araç, uzman veya plan gerekip gerekmediğini seçer.",
  plan: "Önce görev adımlarını çıkarır; uzmanlar sırayla çalışır ve gereken çıktılar denetlenir.",
  supervisor: "Supervisor her adımda uygun uzmanı seçer ve sonucu toplar.",
  router: "İsteği doğrudan uygun uzmana yönlendirir; daha kısa bir akış kullanır.",
  single: "Tek agent kendi izinli araçlarını kullanarak görevi tamamlar.",
  graph: "Aynı agent akışını LangGraph motoruyla yürütür. Geliştirici kullanımı içindir."
};
const chatHistory = document.getElementById("chat-history");
const agentLabels = {
  supervisor: "Supervisor", general: "Genel agent", researcher: "Araştırmacı",
  coder: "Kod agent'ı", file_agent: "Dosya agent'ı", data_agent: "Veri agent'ı",
  writer: "Rapor yazarı", reviewer: "Reviewer", router: "Router", planner: "Planlayıcı",
  single: "Tek agent", mode_router: "Mod yönlendiricisi"
};
const toolLabels = {
  wikipedia_lookup: "Wikipedia araştırması", search: "Belge arama",
  file_read: "Dosya okuma", file_write: "Dosya yazma",
  directory_list: "Klasör listeleme", calculator: "Hesap makinesi",
  csv_summary: "CSV analizi", function_test: "Fonksiyon testi", web_search: "Web araştırması",
  python_exec: "Python çalıştırma"
};
const modeLabels = {
  auto: "Otomatik seçim", plan: "Adımlı görev", supervisor: "Uzman ekip",
  router: "Hızlı router", single: "Tek agent", graph: "LangGraph motoru"
};
const stateLabels = {
  active: "Çalışıyor", done: "Tamamlandı", waiting: "Sırada",
  skipped: "Gerekmedi", failed: "Hata", approval: "Onay bekliyor"
};

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function agentIsActive(view, agent) {
  return view.status === "running" && currentAgent(view) === agent;
}

function currentAgent(view) {
  const stack = [];
  for (const event of view.events) {
    if (event.event === "agent_enter" && event.agent) stack.push(event.agent);
    if (event.event === "agent_exit" && event.agent) {
      const index = stack.lastIndexOf(event.agent);
      if (index >= 0) stack.splice(index, 1);
    }
  }
  return stack.at(-1) || null;
}

function makeFlowNode(id, parents, level, label, detail, state, explanation) {
  return { id, parents: parents ? [].concat(parents) : [], level, label, detail, state, explanation };
}

function makeFlowCard(node) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "flow-card";
  button.dataset.nodeId = node.id;
  button.dataset.state = node.state;
  button.setAttribute("aria-pressed", String(selectedFlowNode === node.id));
  button.setAttribute("aria-label", `${node.label}. ${stateLabels[node.state]}. ${node.detail}`);
  const icon = document.createElement("span");
  icon.className = "flow-card-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = node.state === "done" ? "✓" : node.state === "failed" ? "!"
    : node.id === "answer" && node.state === "active" ? "" : "·";
  const title = document.createElement("strong");
  title.textContent = node.label;
  const state = document.createElement("small");
  state.className = "flow-card-state";
  state.textContent = node.id === "answer" && node.state === "active"
    ? "Yanıt hazırlanıyor…" : stateLabels[node.state] || "Sırada";
  const detail = document.createElement("span");
  detail.className = "flow-card-detail";
  detail.textContent = node.detail;
  button.append(icon, title, state, detail);
  button.addEventListener("click", () => {
    selectedFlowNode = node.id;
    flowGraph.querySelectorAll(".flow-card").forEach((card) => {
      card.setAttribute("aria-pressed", String(card.dataset.nodeId === selectedFlowNode));
    });
    showFlowNode(node);
  });
  return button;
}

function showFlowNode(node) {
  const title = document.createElement("strong");
  title.textContent = `${node.label} · ${stateLabels[node.state] || "Sırada"}`;
  const description = document.createElement("p");
  description.textContent = node.explanation;
  const detail = document.createElement("small");
  detail.textContent = node.detail;
  flowInspector.replaceChildren(title, description, detail);
}

function renderTreeEdges(canvas, nodes, positions, width, height) {
  const svgNs = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNs, "svg");
  svg.classList.add("flow-tree-edges");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("aria-hidden", "true");
  const defs = document.createElementNS(svgNs, "defs");
  for (const [state, color] of Object.entries({
    waiting: "#91a9b1", done: "#27865f", active: "#087f91", failed: "#b94d4d", approval: "#bd7e13"
  })) {
    const marker = document.createElementNS(svgNs, "marker");
    marker.setAttribute("id", `flow-arrow-${state}`);
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "10");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "9");
    marker.setAttribute("markerHeight", "9");
    marker.setAttribute("orient", "auto");
    const tip = document.createElementNS(svgNs, "path");
    tip.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    tip.setAttribute("fill", color);
    marker.append(tip);
    defs.append(marker);
  }
  svg.append(defs);
  for (const node of nodes) {
    const child = positions.get(node.id);
    if (!child) continue;
    for (const parentId of node.parents) {
      const parent = positions.get(parentId);
      if (!parent) continue;
      const path = document.createElementNS(svgNs, "path");
      const startY = parent.y + 60;
      const endY = child.y - 60;
      const curve = Math.max(28, (endY - startY) * 0.42);
      path.setAttribute("d", `M ${parent.x} ${startY} C ${parent.x} ${startY + curve}, ${child.x} ${endY - curve}, ${child.x} ${endY}`);
      path.dataset.state = node.state === "active" ? "active"
        : node.state === "failed" ? "failed"
        : node.state === "approval" ? "approval"
          : node.state === "done" && ["done", "skipped"].includes(parent.state) ? "done" : "waiting";
      path.setAttribute("marker-end", `url(#flow-arrow-${path.dataset.state})`);
      svg.append(path);
    }
  }
  canvas.append(svg);
}

function renderFlowGraph(view) {
  const failed = view.status === "failed";
  const finished = view.status === "completed";
  const running = view.status === "running" || view.status === "waiting_approval";
  const managerNames = ["supervisor", "router", "planner", "mode_router"];
  const activeAgent = view.current_agent || currentAgent(view);
  const systemAgents = new Set([...managerNames, "reviewer"]);
  const managerEvents = view.events.filter((event) => managerNames.includes(event.agent)
    || ["route_selected", "mode_selected", "plan_fallback"].includes(event.event));
  const lastManagerError = [...view.events].reverse().find((event) =>
    event.event === "agent_error" && managerNames.includes(event.agent)
  );
  const lastRecovery = [...view.events].reverse().find((event) => event.event === "supervisor_fallback");
  const managerFailed = Boolean(lastManagerError
    && (!lastRecovery || lastRecovery.sequence < lastManagerError.sequence));
  const managerActive = managerNames.includes(activeAgent) && view.status === "running";
  const selectedMode = [...view.events].reverse().find((event) => event.event === "mode_selected")?.mode;
  const managerLabel = managerEvents.some((event) => ["supervisor", "planner"].includes(event.agent))
    ? "Supervisor / planlayıcı" : "Mod yönlendiricisi";
  let managerDetail = view.route
    ? `${agentLabels[view.route.selected_agent] || view.route.selected_agent} seçildi`
    : modeLabels[selectedMode] || modeLabels[view.mode] || view.mode;
  const managerState = managerFailed ? "failed" : managerActive ? "active"
    : managerEvents.length || view.plan || view.route || view.agent_outputs.length ? "done"
      : running && view.mode !== "single" ? "active" : finished && view.mode === "single" ? "skipped" : "waiting";

  const workerSpecs = view.plan?.tasks?.length
    ? view.plan.tasks.map((task) => ({ id: `worker-${task.id}`, agent: task.agent, task: task.task, plannedId: task.id }))
    : unique([
      ...view.agent_outputs.map((output) => output.agent),
      ...view.events.map((event) => event.agent).filter((agent) => agent && !systemAgents.has(agent)),
      view.route?.selected_agent === "supervisor" ? null : view.route?.selected_agent,
      view.mode === "single" && view.events.length ? "single" : null
    ]).map((agent, index) => ({ id: `worker-${index}`, agent, task: null, plannedId: null }));
  const workers = workerSpecs.map((worker) => {
    const output = view.agent_outputs.find((entry) => worker.plannedId
      ? entry.planned_id === worker.plannedId : entry.agent === worker.agent);
    const hasError = view.events.some((event) => event.event === "agent_error" && event.agent === worker.agent);
    const state = agentIsActive(view, worker.agent) ? "active"
      : hasError ? "failed" : output ? "done" : finished ? "done" : "waiting";
    return { ...worker, output, state };
  });
  const eventTools = view.events.filter((event) => event.event === "tool_call");
  const toolSpecs = view.tool_calls.length
    ? view.tool_calls.map((call, index) => ({ tool: call.tool, state: call.result.success ? "done" : "failed", index }))
    : eventTools.map((event, index) => ({ tool: event.tool, state: event.success === false ? "failed" : "done", index }));
  const tools = toolSpecs.slice(0, 12).map((tool, index) => ({
    id: `tool-${index}`, tool: tool.tool, state: tool.state,
    detail: tool.state === "failed" ? "Araç hata döndürdü" : "Araç sonucu alındı"
  }));
  if (view.pending_approval && !tools.some((tool) => tool.tool === view.pending_approval.tool)) {
    tools.push({
      id: `tool-approval-${tools.length}`, tool: view.pending_approval.tool,
      state: "approval", detail: "Onay yanıtın bekleniyor"
    });
  }
  const reviewEvents = view.events.filter((event) => event.event === "review_verdict");
  const reviewSpecs = view.reviews.length ? view.reviews : reviewEvents.map((event, index) => ({
    agent: "reviewer", attempt: index + 1, status: event.success === false ? "fail" : "pass", task: "Sonucu kontrol etti", issues: []
  }));
  if (agentIsActive(view, "reviewer")) {
    reviewSpecs.push({ agent: "reviewer", attempt: reviewSpecs.length + 1, status: "active", task: "Çıktıyı inceliyor", issues: [] });
  }
  const reviews = reviewSpecs.map((review, index) => ({
    id: `review-${index}`, label: `Reviewer · ${review.attempt}. kontrol`,
    state: review.status === "active" ? "active" : review.status === "pass" ? "done" : "failed",
    detail: review.status === "active" ? "Çıktıyı inceliyor"
      : review.status === "pass" ? "Kontrolden geçti" : "Düzeltme istedi",
    explanation: review.issues?.length ? `Kontrol notu: ${review.issues.join("; ")}` : "Reviewer, agent çıktısını ve varsa araç kanıtlarını kontrol eder."
  }));
  const hasWorkOutput = view.agent_outputs.length > 0 || view.tool_calls.length > 0
    || eventTools.length > 0 || view.reviews.length > 0 || reviewEvents.length > 0
    || view.events.some((event) => event.event === "agent_exit"
      && event.agent && !systemAgents.has(event.agent));
  // Final synthesis runs outside an agent span, so an empty active-agent stack
  // after worker output is the expected answer-generation phase.
  const answerIsActive = running && hasWorkOutput && !view.pending_approval
    && (managerActive || !activeAgent);
  if (answerIsActive && managerActive) managerDetail = "Son yanıtı toparlıyor";
  const answerState = finished ? "done" : failed ? "failed" : answerIsActive ? "active" : "waiting";
  const nodes = [];
  const request = makeFlowNode("request", null, 0, "Senin isteğin",
    view.message.length > 72 ? `${view.message.slice(0, 72)}…` : view.message,
    "done", "Mesajın ve bu konuşmanın geçmişi görev girdisi olarak alınır.");
  const manager = makeFlowNode("manager", "request", 1, managerLabel, managerDetail,
    managerState, "İsteğin türüne göre doğrudan sohbeti sürdürür, bir agent seçer veya işi uzmanlara böler.");
  nodes.push(request, manager);
  for (const [index, worker] of workers.entries()) {
    const task = worker.task || worker.output?.task || "Görev için seçilen uzman";
    nodes.push(makeFlowNode(worker.id, "manager", 2,
      agentLabels[worker.agent] || worker.agent, task, worker.state,
      `Bu agent kendisine verilen alt görevi yapar.${worker.output?.answer ? ` Çıktı: ${worker.output.answer.slice(0, 180)}` : ""}`));
  }
  for (const tool of tools) {
    nodes.push(makeFlowNode(tool.id, "manager", 2,
      toolLabels[tool.tool] || tool.tool, tool.detail, tool.state,
      "Agent bu aracı kullanarak hesaplama, dosya işlemi veya kaynak araştırması yapar."));
  }
  let branchLeaves = [...workers.map((worker) => worker.id), ...tools.map((tool) => tool.id)];
  if (!branchLeaves.length) {
    const idleState = finished ? "skipped" : failed ? "failed" : "waiting";
    nodes.push(makeFlowNode("no-specialist", "manager", 2, "Ek agent veya araç yok",
      "Bu görevde uzman/araç gerekmedi", idleState,
      "Basit sohbet isteklerinde sistem doğrudan yanıt verebilir."));
    branchLeaves = ["no-specialist"];
  }
  let previousReviewIds = branchLeaves;
  for (const review of reviews) {
    nodes.push(makeFlowNode(review.id, previousReviewIds, 3, review.label,
      review.detail, review.state, review.explanation));
    previousReviewIds = [review.id];
  }
  if (!reviews.length) {
    nodes.push(makeFlowNode("review-skipped", branchLeaves, 3, "Reviewer",
      finished || answerIsActive ? "İnceleme gerekmedi" : "İnceleme bekleniyor",
      finished || answerIsActive ? "skipped" : "waiting",
      "Kod veya araştırma gibi görevlerde reviewer çıktıyı doğrulayabilir."));
  }
  nodes.push(makeFlowNode("answer", reviews.length ? reviews.at(-1).id : "review-skipped", 4,
    finished ? "Yanıt hazır" : failed ? "Görev durdu" : "Yanıt hazırlanıyor",
    finished ? "Sonuç sohbete eklendi" : failed ? view.error?.code || "Görev tamamlanamadı"
      : answerIsActive ? "Supervisor son yanıtı hazırlıyor" : "Önceki adımlar bekleniyor",
    answerState, "Supervisor agent ve araç sonuçlarını birleştirip son yanıtı konuşmaya ekler."));

  const viewportWidth = Math.max(flowGraph.clientWidth, 760);
  const canvasWidth = Math.max(viewportWidth, (Math.max(workers.length + tools.length, 1) * 236) + 80);
  const rows = [60, 225, 400, 580, 760];
  const levelCounts = new Map();
  for (const node of nodes) levelCounts.set(node.level, (levelCounts.get(node.level) || 0) + 1);
  const levelIndexes = new Map();
  const positions = new Map();
  for (const node of nodes) {
    const count = levelCounts.get(node.level);
    const index = levelIndexes.get(node.level) || 0;
    levelIndexes.set(node.level, index + 1);
    const x = canvasWidth * ((index + 0.5) / count);
    const y = rows[node.level] ?? rows.at(-1);
    positions.set(node.id, { x, y, state: node.state });
  }
  flowGraph.replaceChildren();
  const viewport = document.createElement("div");
  viewport.className = "flow-tree-viewport";
  viewport.setAttribute("tabindex", "0");
  viewport.setAttribute("aria-label", "Görev ağacı. Geniş görünümde yana kaydırılabilir.");
  const canvas = document.createElement("div");
  canvas.className = "flow-tree-canvas";
  canvas.style.width = `${canvasWidth}px`;
  canvas.style.height = "840px";
  renderTreeEdges(canvas, nodes, positions, canvasWidth, 840);
  for (const node of nodes) {
    const card = makeFlowCard(node);
    const position = positions.get(node.id);
    card.style.left = `${position.x}px`;
    card.style.top = `${position.y}px`;
    canvas.append(card);
  }
  viewport.append(canvas);
  flowGraph.append(viewport);
  const currentSelection = nodes.find((node) => node.id === selectedFlowNode)
    || nodes.find((node) => node.id === "answer" && node.state === "active")
    || nodes.find((node) => node.state === "active") || nodes.at(-1);
  if (currentSelection) showFlowNode(currentSelection);

  const runningStage = answerIsActive ? "Yanıt hazırlanıyor"
    : activeAgent === "reviewer" ? "Reviewer sonucu kontrol ediyor"
      : managerActive ? "Supervisor işi yönlendiriyor"
        : activeAgent ? `${agentLabels[activeAgent] || activeAgent} çalışıyor`
          : view.status === "queued" ? "Görev sırada bekliyor" : "Görev akışı başlatılıyor";
  const completedCount = nodes.filter((node) => node.state === "done").length;
  const activeDetail = answerIsActive ? "Son yanıt hazırlanıyor; önceki görev adımları tamamlandı."
    : activeAgent === "reviewer" ? "Çalışan kart ve turkuaz ok, şu an kontrol edilen adımı gösteriyor."
      : activeAgent && !managerNames.includes(activeAgent)
        ? `${agentLabels[activeAgent] || activeAgent} görevi yürütüyor; turkuaz ok bu adıma gider.`
        : "Turkuaz oklar şu an izlenen yolu, yeşil oklar tamamlanan adımları gösterir.";
  flowSummary.textContent = failed ? `Durdu: ${view.error?.code || "görev hatası"}. Hata veren adım kırmızı.`
    : finished ? "Tamamlandı: Son yanıt sohbete eklendi."
      : view.status === "waiting_approval" ? "Onay bekliyor: Dosya işlemi için yanıtın gerekiyor."
        : `Şu an: ${runningStage}. ${activeDetail} (${completedCount} adım tamamlandı).`;
  flowGraph.setAttribute("aria-label", `Agent görev akışı. ${flowSummary.textContent}`);
}

function showMessages(messages) {
  if (!messages.length) return empty(chatHistory, "İlk mesajını yaz.");
  chatHistory.replaceChildren(...messages.map((message) => {
    const bubble = document.createElement("div");
    bubble.className = `chat-message ${message.role}`;
    const label = document.createElement("strong");
    label.textContent = message.role === "user" ? "Sen" : "Asistan";
    const body = document.createElement("p");
    body.textContent = message.content;
    bubble.append(label, body);
    return bubble;
  }));
  chatHistory.lastElementChild?.scrollIntoView({ block: "nearest" });
}

async function loadConversation() {
  const response = await fetch(`/conversations/${conversationId}`);
  if (!response.ok) throw new Error("Sohbet geçmişi yüklenemedi.");
  const data = await response.json();
  showMessages(data.messages);
}

loadConversation().catch((error) => showError(error.message));

const memoryForm = document.getElementById("memory-form");
const memoryList = document.getElementById("memory-list");
const memoryError = document.getElementById("memory-error");

async function loadMemories() {
  const response = await fetch("/memories");
  if (!response.ok) throw new Error("Bellek yüklenemedi.");
  const data = await response.json();
  if (!data.memories.length) return empty(memoryList, "Henüz kalıcı bellek kaydı yok.");
  memoryList.replaceChildren(...data.memories.map((entry) => {
    const row = document.createElement("article");
    row.className = "memory-entry";
    const copy = document.createElement("div");
    const title = document.createElement("strong");
    const category = {preference:"Tercih",decision:"Karar",fact:"Bilgi"}[entry.category];
    title.textContent = `${entry.key} · ${category}`;
    const value = document.createElement("p");
    value.textContent = entry.value;
    copy.append(title, value);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary";
    remove.textContent = "Sil";
    remove.addEventListener("click", async () => {
      const deleted = await fetch(`/memories/${encodeURIComponent(entry.key)}`, { method: "DELETE" });
      if (!deleted.ok) { memoryError.textContent = "Bellek silinemedi."; memoryError.hidden = false; return; }
      loadMemories().catch((error) => { memoryError.textContent = error.message; memoryError.hidden = false; });
    });
    row.append(copy, remove);
    return row;
  }));
}

loadMemories().catch((error) => { memoryError.textContent = error.message; memoryError.hidden = false; });
memoryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  memoryError.hidden = true;
  const payload = Object.fromEntries(new FormData(memoryForm));
  try {
    const response = await fetch("/memories", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!response.ok) throw new Error(response.status === 422
      ? "Anahtar küçük harfle başlamalı (ör. name); hatırlanacak bilgi boş olmamalı. Parola veya gizli anahtar kaydedilemez."
      : "Bellek kaydedilemedi.");
    memoryForm.reset();
    await loadMemories();
  } catch (error) { memoryError.textContent = error.message; memoryError.hidden = false; }
});

document.getElementById("new-chat-button").addEventListener("click", async () => {
  if (activeTask && runButton.disabled) return;
  try {
    const response = await fetch(`/conversations/${conversationId}`, { method: "DELETE" });
    if (!response.ok) throw new Error("Sohbet silinemedi.");
    conversationId = crypto.randomUUID();
    localStorage.setItem("agent-conversation-id", conversationId);
    loadedTask = null;
    activeTask = null;
    showMessages([]);
    showError("");
  } catch (error) { showError(error.message); }
});
const modeSelect = document.getElementById("task-mode");
modeSelect.addEventListener("change", () => {
  document.getElementById("mode-description").textContent = modeDescriptions[modeSelect.value];
});

const labels = {
  queued: "Kuyrukta", running: "Çalışıyor", waiting_approval: "Onay bekliyor",
  completed: "Tamamlandı", failed: "Başarısız"
};
const eventLabels = {
  task_started: "Görev başladı", task_completed: "Görev tamamlandı",
  task_failed: "Görev başarısız", agent_enter: "Agent çalışmaya başladı",
  agent_exit: "Agent tamamladı", agent_error: "Agent hatası",
  model_call: "Model yanıt verdi", model_error: "Model hatası",
  model_retry: "Model yeniden denendi", tool_call: "Araç çalıştı",
  plan_fallback: "Plan üretilemedi; supervisor devam ediyor",
  review_verdict: "Reviewer karar verdi", route_selected: "Agent seçildi",
  mode_selected: "Çalışma biçimi seçildi",
  approval_requested: "Onay istendi", approval_resolved: "Onay yanıtlandı",
  supervisor_fallback: "Agent çıktısından yedek yanıt oluşturuldu",
  context_fallback: "Bağlam sınırı için uygun uzman seçildi"
};

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = !message;
}

function empty(container, text) {
  const p = document.createElement("p");
  p.className = "empty";
  p.textContent = text;
  container.replaceChildren(p);
}

function detailItem(title, subtitle, detail) {
  const item = document.createElement("div");
  item.className = "result-item";
  const strong = document.createElement("strong");
  strong.textContent = title;
  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = subtitle;
  item.append(strong, meta);
  if (detail) {
    const section = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Girdiyi ve sonucu gör";
    const pre = document.createElement("pre");
    pre.textContent = detail;
    section.append(summary, pre);
    item.append(section);
  }
  return item;
}

function render(view) {
  activeTask = view.task_id;
  conversationId = view.conversation_id;
  localStorage.setItem("agent-conversation-id", conversationId);
  document.getElementById("task-id").textContent = `Görev: ${view.task_id}`;
  statusEl.textContent = labels[view.status] || view.status;
  statusEl.dataset.state = view.status;
  runButton.disabled = !["completed", "failed"].includes(view.status);
  renderFlowGraph(view);
  if (view.error) showError(`${view.error.code}: ${view.error.message}`);
  else showError("");

  const rows = view.events.map((event) => {
    const row = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = `${eventLabels[event.event] || event.event}${event.agent ? ` · ${event.agent}` : ""}${event.tool ? ` · ${event.tool}` : ""}`;
    const details = document.createElement("small");
    const parts = [new Date(event.timestamp).toLocaleTimeString("tr-TR")];
    if (event.mode) parts.push(modeLabels[event.mode] || event.mode);
    if (event.duration_ms !== undefined) parts.push(`${Math.round(event.duration_ms)} ms`);
    if (event.prompt_tokens !== undefined) parts.push(`${event.prompt_tokens} giriş tokenı`);
    if (event.completion_tokens !== undefined) parts.push(`${event.completion_tokens} çıkış tokenı`);
    if (event.error_type) parts.push(event.error_type);
    details.textContent = parts.join(" · ");
    row.append(title, details);
    return row;
  });
  if (rows.length) timeline.replaceChildren(...rows);
  else empty(timeline, "Görev kuyruğa alındı.");

  const agentItems = [];
  if (view.route) agentItems.push(detailItem(
    "Router kararı", `Seçilen: ${view.route.selected_agent}`,
    JSON.stringify(view.route, null, 2)
  ));
  if (view.plan) agentItems.push(detailItem(
    "Görev planı", `${view.plan.tasks.length} adım`,
    view.plan.tasks.map((task) =>
      `${task.id}. ${task.agent}: ${task.task}${task.depends_on.length ? ` (bağımlı: ${task.depends_on.join(", ")})` : ""}`
    ).join("\n")
  ));
  for (const output of view.agent_outputs) agentItems.push(detailItem(
    output.agent, output.task, output.answer
  ));
  if (agentItems.length) agentsEl.replaceChildren(...agentItems);
  else empty(agentsEl, "Henüz agent çıktısı yok.");

  if (view.tool_calls.length) {
    toolsEl.replaceChildren(...view.tool_calls.map((call) => detailItem(
      call.tool, call.result.success ? "Başarılı" : `Hata: ${call.result.error_type || "Bilinmiyor"}`,
      JSON.stringify({ arguments: call.arguments, result: call.result }, null, 2)
    )));
  } else empty(toolsEl, "Henüz araç kullanılmadı.");
  if (view.reviews.length) {
    reviewsEl.replaceChildren(...view.reviews.map((review) => detailItem(
      `${review.agent} · ${review.status === "pass" ? "Geçti" : "Düzeltme istendi"}`,
      `Deneme ${review.attempt} · ${review.task}`,
      review.issues.join("\n")
    )));
  } else empty(reviewsEl, "Henüz inceleme yok.");

  approvalEl.hidden = !view.pending_approval;
  if (view.pending_approval) {
    approvalPreview.textContent = JSON.stringify(view.pending_approval, null, 2);
    document.getElementById("approve-button").disabled = false;
    document.getElementById("deny-button").disabled = false;
  }
  answerEl.textContent = view.answer || "Görev tamamlandığında burada görünecek.";
  answerEl.className = view.answer ? "" : "empty";
  if (["completed", "failed"].includes(view.status) && loadedTask !== view.task_id) {
    loadedTask = view.task_id;
    loadConversation().catch((error) => showError(error.message));
  }
  if (["completed", "failed"].includes(view.status) && stream) {
    stream.close();
    stream = null;
  }
}

async function sendApproval(approved) {
  if (!activeTask) return;
  document.getElementById("approve-button").disabled = true;
  document.getElementById("deny-button").disabled = true;
  try {
    const response = await fetch(`/tasks/${activeTask}/approval`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved })
    });
    if (!response.ok) throw new Error(`Onay yanıtı gönderilemedi (${response.status}).`);
    render(await response.json());
  } catch (error) {
    showError(error.message);
    document.getElementById("approve-button").disabled = false;
    document.getElementById("deny-button").disabled = false;
  }
}

document.getElementById("approve-button").addEventListener("click", () => sendApproval(true));
document.getElementById("deny-button").addEventListener("click", () => sendApproval(false));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (stream) stream.close();
  runButton.disabled = true;
  showError("");
  const message = document.getElementById("task-message").value.trim();
  if (!message) { runButton.disabled = false; return; }
  const pending = [...chatHistory.querySelectorAll(".chat-message")].map((item) => ({
    role: item.classList.contains("user") ? "user" : "assistant",
    content: item.querySelector("p").textContent
  }));
  showMessages([...pending, { role: "user", content: message }]);
  try {
    const response = await fetch("/tasks", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        mode: modeSelect.value,
        conversation_id: conversationId
      })
    });
    if (!response.ok) throw new Error(`Görev başlatılamadı (${response.status}).`);
    const view = await response.json();
    document.getElementById("task-message").value = "";
    render(view);
    stream = new EventSource(`/tasks/${view.task_id}/events`);
    stream.addEventListener("update", (update) => render(JSON.parse(update.data)));
    stream.onerror = () => showError("Olay bağlantısı kesildi. Yeniden bağlanılıyor.");
  } catch (error) {
    runButton.disabled = false;
    showError(error.message);
    loadConversation().catch(() => {});
  }
});
