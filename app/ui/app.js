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
let stream = null;
let activeTask = null;
let conversationId = localStorage.getItem("agent-conversation-id") || crypto.randomUUID();
let loadedTask = null;
localStorage.setItem("agent-conversation-id", conversationId);
const modeDescriptions = {
  auto: "Yerel model isteğe ve sohbet geçmişine bakarak sohbet, tek agent, supervisor veya planlı akışı seçer.",
  plan: "Görevi adımlara ayırır, uzman agent'lara verir ve yazılan dosyaları reviewer ile kontrol eder.",
  supervisor: "Supervisor her turda sıradaki agent'ı seçer. Çok adımlı görevler için deneysel alternatiftir.",
  router: "Önce basit görevleri tek uzmana yönlendirir; karmaşık işlerde supervisor'a geçer.",
  single: "Tek agent kendi araçlarını kullanır; uzmanlar arasında görev dağıtmaz.",
  graph: "Aynı yerel model ve uzmanları LangGraph akış motoruyla çalıştırır."
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
  csv_summary: "CSV analizi", function_test: "Fonksiyon testi",
  python_exec: "Python çalıştırma"
};
const modeLabels = {
  auto: "Otomatik seçim", plan: "Planlı supervisor", supervisor: "Supervisor",
  router: "Hızlı router", single: "Tek agent", graph: "LangGraph motoru"
};
const stateLabels = {
  active: "Çalışıyor", done: "Tamamlandı", waiting: "Sırada",
  skipped: "Gerekmedi", failed: "Hata", approval: "Onay bekliyor"
};

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function latestEvent(view, names) {
  return [...view.events].reverse().find((event) => names.includes(event.event));
}

function agentIsActive(view, agent) {
  const events = view.events.filter((event) => event.agent === agent);
  const last = events.at(-1);
  return last?.event === "agent_enter" && view.status === "running";
}

function graphItem(label, detail, state = "done") {
  return { label, detail, state };
}

function makeFlowStage(stage) {
  const article = document.createElement("article");
  article.className = "flow-stage";
  article.dataset.state = stage.state;
  const head = document.createElement("div");
  head.className = "flow-stage-head";
  const number = document.createElement("span");
  number.className = "flow-number";
  number.textContent = stage.number;
  const titleWrap = document.createElement("div");
  const title = document.createElement("h4");
  title.textContent = stage.title;
  const status = document.createElement("span");
  status.className = "flow-state";
  status.textContent = stateLabels[stage.state];
  titleWrap.append(title, status);
  head.append(number, titleWrap);
  const caption = document.createElement("p");
  caption.className = "flow-caption";
  caption.textContent = stage.caption;
  const items = document.createElement("div");
  items.className = "flow-node-list";
  for (const entry of stage.items) {
    const item = document.createElement("div");
    item.className = "flow-node";
    item.dataset.state = entry.state;
    const dot = document.createElement("i");
    dot.setAttribute("aria-hidden", "true");
    const copy = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = entry.label;
    copy.append(strong);
    if (entry.detail) {
      const small = document.createElement("small");
      small.textContent = entry.detail;
      copy.append(small);
    }
    item.append(dot, copy);
    items.append(item);
  }
  article.append(head, caption, items);
  return article;
}

function renderFlowGraph(view) {
  const failed = view.status === "failed";
  const finished = view.status === "completed";
  const running = view.status === "running" || view.status === "waiting_approval";
  const managerEvents = view.events.filter((event) =>
    ["supervisor", "router", "planner", "mode_router"].includes(event.agent)
    || ["route_selected", "mode_selected", "plan_fallback"].includes(event.event)
  );
  const systemAgents = new Set(["supervisor", "router", "planner", "mode_router", "reviewer"]);
  const workerNames = unique([
    ...(view.plan?.tasks || []).map((task) => task.agent),
    ...view.agent_outputs.map((output) => output.agent),
    ...view.events.map((event) => event.agent).filter((agent) => agent && !systemAgents.has(agent)),
    view.route?.selected_agent === "supervisor" ? null : view.route?.selected_agent
  ]);
  if (view.mode === "single" && !workerNames.length && view.events.length) workerNames.push("single");
  const eventTools = view.events.filter((event) => event.event === "tool_call");
  const toolNames = unique([...view.tool_calls.map((call) => call.tool), ...eventTools.map((e) => e.tool)]);
  const reviewEvents = view.events.filter((event) => event.event === "review_verdict");
  const hasReview = view.reviews.length > 0 || reviewEvents.length > 0;
  const managerFailed = view.events.some((event) =>
    event.event === "agent_error" && ["supervisor", "router", "planner", "mode_router"].includes(event.agent)
  );
  const workerFailed = view.events.some((event) =>
    event.event === "agent_error" && event.agent && !systemAgents.has(event.agent)
  );
  const toolFailed = view.tool_calls.some((call) => !call.result.success)
    || eventTools.some((event) => event.success === false);
  const activeWorker = workerNames.some((agent) => agentIsActive(view, agent));
  const managerActive = ["supervisor", "router", "planner", "mode_router"].some((agent) =>
    agentIsActive(view, agent)
  );
  const directSingle = workerNames.includes("single")
    && !managerEvents.some((event) => event.agent === "supervisor" || event.agent === "planner");

  let managerState = "waiting";
  if (managerFailed) managerState = "failed";
  else if (managerActive || (running && managerEvents.length === 0
      && view.mode !== "single" && !directSingle)) {
    managerState = "active";
  } else if (managerEvents.length || view.plan || view.route || view.agent_outputs.length) {
    managerState = "done";
  } else if (finished || view.mode === "single" || directSingle) managerState = "skipped";

  let workerState = "waiting";
  if (workerFailed) workerState = "failed";
  else if (activeWorker) workerState = "active";
  else if (workerNames.length && (view.agent_outputs.length || finished || failed)) workerState = "done";
  else if (finished && !workerNames.length) workerState = "skipped";

  let toolState = "waiting";
  if (toolFailed) toolState = "failed";
  else if (view.status === "waiting_approval") toolState = "approval";
  else if (toolNames.length) toolState = "done";
  else if (finished) toolState = "skipped";

  const reviewState = hasReview
    ? (view.reviews.some((review) => review.status === "fail") ? "failed" : "done")
    : (finished ? "skipped" : "waiting");
  const answerState = finished ? "done" : failed ? "failed" : "waiting";
  const selectedMode = [...view.events].reverse().find((event) => event.event === "mode_selected")?.mode;
  const managerDetail = view.route
    ? `${agentLabels[view.route.selected_agent] || view.route.selected_agent} seçildi`
    : modeLabels[selectedMode] || modeLabels[view.mode] || view.mode;

  const stages = [
    {
      number: "01", title: "İstek", state: failed ? "failed" : "done",
      caption: "Mesaj ve konuşma geçmişi alınır.",
      items: [graphItem("Senin mesajın", view.message.length > 54 ? `${view.message.slice(0, 54)}…` : view.message)]
    },
    {
      number: "02", title: "Yönetici", state: managerState,
      caption: "Görevin nasıl yürütüleceğine karar verir.",
      items: [graphItem(
        managerEvents.some((event) => ["supervisor", "planner"].includes(event.agent))
          ? "Supervisor / Planlayıcı" : "Mod yönlendiricisi",
        managerDetail, managerState
      )]
    },
    {
      number: "03", title: "Uzmanlar", state: workerState,
      caption: "İşi uygun uzman agent gerçekleştirir.",
      items: workerNames.length ? workerNames.map((name) => graphItem(
        agentLabels[name] || name,
        view.agent_outputs.find((output) => output.agent === name)?.task || "Atanan görev",
        agentIsActive(view, name) ? "active" : workerFailed ? "failed" : "done"
      )) : [graphItem("Uzman agent", "Bu görevde uzman seçilmedi.", workerState)]
    },
    {
      number: "04", title: "Araçlar", state: toolState,
      caption: "Dosya, hesaplama veya araştırma araçları çalışır.",
      items: toolNames.length ? toolNames.map((name) => {
        const call = view.tool_calls.find((entry) => entry.tool === name);
        const event = [...eventTools].reverse().find((entry) => entry.tool === name);
        const success = call ? call.result.success : event?.success !== false;
        return graphItem(toolLabels[name] || name, success ? "Başarılı" : "Araç hatası", success ? "done" : "failed");
      }) : [graphItem("Araç çağrısı", "Bu görevde araç kullanılmadı.", toolState)]
    },
    {
      number: "05", title: "Kontrol", state: reviewState,
      caption: "Gerekli görevlerde reviewer sonucu doğrular.",
      items: hasReview ? view.reviews.map((review) => graphItem(
        "Reviewer", review.status === "pass" ? "Kontrolden geçti" : "Düzeltme istedi",
        review.status === "pass" ? "done" : "failed"
      )) : [graphItem("Reviewer", "Bu görevde inceleme gerekmedi.", reviewState)]
    },
    {
      number: "06", title: "Yanıt", state: answerState,
      caption: "Toplanan sonuç tek yanıta dönüştürülür.",
      items: [graphItem(
        finished ? "Yanıt hazır" : failed ? "Görev tamamlanamadı" : "Yanıt hazırlanıyor",
        finished ? "Sohbete eklendi" : failed ? view.error?.code || "Hata" : "Önceki aşamalar bekleniyor",
        answerState
      )]
    }
  ];
  flowGraph.replaceChildren(...stages.map(makeFlowStage));

  const last = latestEvent(view, [
    "task_failed", "task_completed", "approval_requested", "review_verdict",
    "tool_call", "agent_enter", "route_selected", "mode_selected", "task_started"
  ]);
  const activeLabel = last?.agent ? agentLabels[last.agent] || last.agent : null;
  const activeTool = last?.tool ? toolLabels[last.tool] || last.tool : null;
  const summaries = {
    task_started: "İstek alındı; görev yolu belirleniyor.",
    route_selected: `${activeLabel || "Uzman agent"} seçildi.`,
    mode_selected: `${modeLabels[last?.mode] || "Çalışma biçimi"} seçildi.`,
    agent_enter: `${activeLabel || "Agent"} şu anda çalışıyor.`,
    tool_call: `${activeTool || "Araç"} çalışmasını tamamladı.`,
    review_verdict: "Reviewer sonucu kontrol etti.",
    approval_requested: "Dosya işlemi için onayın bekleniyor.",
    task_completed: "Akış tamamlandı; yanıt sohbete eklendi.",
    task_failed: `Akış ${view.error?.code || "bir hata"} nedeniyle durdu.`
  };
  flowSummary.textContent = summaries[last?.event] || (view.status === "queued"
    ? "Görev sıraya alındı." : "Görev akışı hazırlanıyor.");
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
  approval_requested: "Onay istendi", approval_resolved: "Onay yanıtlandı"
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
