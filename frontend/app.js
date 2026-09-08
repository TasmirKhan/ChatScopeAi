// ChatScope AI - frontend logic. No build step, no framework - plain fetch + DOM.

const API = {
  search: "/api/search",
  threads: "/api/threads",
  thread: (id) => `/api/thread/${encodeURIComponent(id)}`,
  stats: "/api/stats",
  senders: "/api/senders",
  topics: "/api/topics",
};

const els = {
  topbarStats: document.getElementById("topbar-stats"),
  form: document.getElementById("search-form"),
  input: document.getElementById("search-input"),
  filterSender: document.getElementById("filter-sender"),
  filterDateFrom: document.getElementById("filter-date-from"),
  filterDateTo: document.getElementById("filter-date-to"),
  filterTopic: document.getElementById("filter-topic"),
  clearFilters: document.getElementById("clear-filters"),
  threadChips: document.getElementById("thread-chips"),
  statusLine: document.getElementById("status-line"),
  results: document.getElementById("results"),
};

// Deterministic pastel-on-dark color per sender, drawn from a small fixed
// palette so avatars stay legible against the dark background.
const AVATAR_COLORS = ["#E8A33D", "#C1546B", "#7FB3A8", "#9B8FD9", "#D98C63", "#7BA6D9", "#C9A24A", "#D97A9C"];
function colorFor(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  return AVATAR_COLORS[hash % AVATAR_COLORS.length];
}
function initials(name) {
  return name.split(" ").map((p) => p[0]).join("").slice(0, 2).toUpperCase();
}

function formatTimestamp(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    day: "numeric", month: "short", year: "numeric",
    hour: "numeric", minute: "2-digit",
  });
}

function setStatus(text, isError = false) {
  els.statusLine.textContent = text;
  els.statusLine.classList.toggle("error", isError);
}

// ---------------------------------------------------------------------------
// Init: stats, senders, topics, decision threads
// ---------------------------------------------------------------------------

async function init() {
  try {
    const stats = await fetchJSON(API.stats);
    els.topbarStats.textContent =
      `${stats.total_messages.toLocaleString()} messages · ${Object.keys(stats.senders).length} people · ` +
      `${stats.date_range.start} to ${stats.date_range.end}`;
  } catch (e) {
    els.topbarStats.textContent = "index not ready yet";
  }

  try {
    const senders = await fetchJSON(API.senders);
    for (const s of senders) {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      els.filterSender.appendChild(opt);
    }
  } catch (e) { /* non-fatal */ }

  try {
    const topics = await fetchJSON(API.topics);
    for (const t of topics) {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      els.filterTopic.appendChild(opt);
    }
  } catch (e) { /* non-fatal */ }

  try {
    const threads = await fetchJSON(API.threads);
    renderThreadChips(threads);
  } catch (e) {
    els.threadChips.innerHTML = "";
  }

  renderEmptyState("Search above, or open a decision thread, to see conversation excerpts here.");
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Decision thread chips
// ---------------------------------------------------------------------------

const THREAD_LABELS = {
  thread_trip_001: "Manali trip",
  thread_project_001: "Group project tech stack",
  thread_dinner_001: "Dinner meetup",
};

function renderThreadChips(threads) {
  els.threadChips.innerHTML = "";
  for (const t of threads) {
    const btn = document.createElement("button");
    btn.className = "thread-chip";
    const label = THREAD_LABELS[t.thread_id] || t.thread_id;
    btn.innerHTML = `<span class="chip-title">${escapeHTML(label)}</span>` +
      `<span class="chip-meta">${t.message_count} messages · ${t.start_date} → ${t.end_date}</span>`;
    btn.addEventListener("click", () => openThread(t.thread_id, label));
    els.threadChips.appendChild(btn);
  }
}

async function openThread(threadId, label) {
  setStatus(`Loading "${label}"…`);
  els.results.innerHTML = "";
  try {
    const data = await fetchJSON(API.thread(threadId));
    setStatus(`${data.message_count} messages in "${label}" — full thread, chronological`);

    const backBtn = document.createElement("button");
    backBtn.className = "back-link";
    backBtn.textContent = "← back to search";
    backBtn.addEventListener("click", () => {
      els.results.innerHTML = "";
      renderEmptyState("Search above, or open a decision thread, to see conversation excerpts here.");
      setStatus("");
    });
    els.results.appendChild(backBtn);

    data.messages.forEach((msg, i) => {
      const isLast = i === data.messages.length - 1;
      els.results.appendChild(renderMessage(msg, { isDecision: isLast, showScore: false }));
    });
  } catch (e) {
    setStatus(e.message, true);
  }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = els.input.value.trim();
  if (!query) return;

  setStatus("Searching…");
  els.results.innerHTML = "";

  const payload = {
    query,
    top_k: 15,
    sender: els.filterSender.value || undefined,
    date_from: els.filterDateFrom.value || undefined,
    date_to: els.filterDateTo.value || undefined,
    topic: els.filterTopic.value || undefined,
  };

  try {
    const data = await fetchJSON(API.search, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (data.count === 0) {
      setStatus(`No matches for "${query}" with the current filters.`);
      renderEmptyState("Try loosening a filter, or rephrasing the query.");
      return;
    }

    setStatus(`${data.count} results for "${query}", ranked by semantic similarity`);
    const maxScore = Math.max(...data.results.map((r) => r.score));
    for (const r of data.results) {
      els.results.appendChild(renderMessage(r, { isDecision: false, showScore: true, maxScore }));
    }
  } catch (e) {
    setStatus(e.message, true);
  }
});

els.clearFilters.addEventListener("click", () => {
  els.filterSender.value = "";
  els.filterDateFrom.value = "";
  els.filterDateTo.value = "";
  els.filterTopic.value = "";
});

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function renderMessage(msg, { isDecision, showScore, maxScore }) {
  const row = document.createElement("div");
  row.className = "msg-row" + (isDecision ? " decision" : "");

  const head = document.createElement("div");
  head.className = "msg-head";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.style.background = colorFor(msg.sender);
  avatar.textContent = initials(msg.sender);

  const name = document.createElement("span");
  name.className = "sender-name";
  name.textContent = msg.sender;

  const meta = document.createElement("span");
  meta.className = "msg-meta";
  meta.textContent = formatTimestamp(msg.timestamp);

  head.appendChild(avatar);
  head.appendChild(name);
  head.appendChild(meta);

  if (isDecision) {
    const tag = document.createElement("span");
    tag.className = "decision-tag";
    tag.textContent = "decision reached";
    head.appendChild(tag);
  }

  const text = document.createElement("p");
  text.className = "msg-text";
  text.textContent = msg.message;

  row.appendChild(head);
  row.appendChild(text);

  if (showScore && typeof msg.score === "number") {
    const track = document.createElement("div");
    track.className = "relevance-track";
    const fill = document.createElement("div");
    fill.className = "relevance-fill";
    const pct = maxScore > 0 ? Math.max(6, (msg.score / maxScore) * 100) : 0;
    fill.style.width = `${pct}%`;
    track.appendChild(fill);
    row.appendChild(track);
  }

  return row;
}

function renderEmptyState(text) {
  const div = document.createElement("div");
  div.className = "empty-state";
  div.textContent = text;
  els.results.appendChild(div);
}

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

init();
