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
let stream = null;
let activeTask = null;
const modeDescriptions = {
  plan: "Görevi adımlara ayırır, uzman agent'lara verir ve yazılan dosyaları reviewer ile kontrol eder.",
  supervisor: "Supervisor her turda sıradaki agent'ı seçer. Çok adımlı görevler için deneysel alternatiftir.",
  router: "Önce basit görevleri tek uzmana yönlendirir; karmaşık işlerde supervisor'a geçer.",
  single: "Tek agent kendi araçlarını kullanır; uzmanlar arasında görev dağıtmaz.",
  graph: "Aynı yerel model ve uzmanları LangGraph akış motoruyla çalıştırır."
};
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
  review_verdict: "Reviewer karar verdi", route_selected: "Agent seçildi",
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
  document.getElementById("task-id").textContent = `Görev: ${view.task_id}`;
  statusEl.textContent = labels[view.status] || view.status;
  statusEl.dataset.state = view.status;
  runButton.disabled = !["completed", "failed"].includes(view.status);
  if (view.error) showError(`${view.error.code}: ${view.error.message}`);
  else showError("");

  const rows = view.events.map((event) => {
    const row = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = `${eventLabels[event.event] || event.event}${event.agent ? ` · ${event.agent}` : ""}${event.tool ? ` · ${event.tool}` : ""}`;
    const details = document.createElement("small");
    const parts = [new Date(event.timestamp).toLocaleTimeString("tr-TR")];
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
  try {
    const response = await fetch("/tasks", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: document.getElementById("task-message").value.trim(),
        mode: modeSelect.value
      })
    });
    if (!response.ok) throw new Error(`Görev başlatılamadı (${response.status}).`);
    const view = await response.json();
    render(view);
    stream = new EventSource(`/tasks/${view.task_id}/events`);
    stream.addEventListener("update", (update) => render(JSON.parse(update.data)));
    stream.onerror = () => showError("Olay bağlantısı kesildi. Yeniden bağlanılıyor.");
  } catch (error) {
    runButton.disabled = false;
    showError(error.message);
  }
});
