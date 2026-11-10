/* Settings: account + usage, BYOK keys, billing. Requires a session. */
(function () {
  "use strict";

  var Sherlock = window.Sherlock;
  var el = Sherlock.el;
  var NEAR_LIMIT_RATIO = 0.8;
  var HIGHLIGHT_MS = 4000;
  var PLAN_LABELS = { free: "Free", pro: "Pro" };
  var CUSTOM_PROVIDER = "custom";

  var params = new URLSearchParams(window.location.search);
  var numberFormat = new Intl.NumberFormat("en-US");
  var providerById = {};

  function $(id) {
    return document.getElementById(id);
  }

  function planLabel(plan) {
    return PLAN_LABELS[plan] || String(plan || "—");
  }

  function formatDate(value) {
    var date = value ? new Date(value) : null;
    if (!date || Number.isNaN(date.getTime())) return "—";
    return date.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  }

  function formatMoney(cents, currency) {
    if (typeof cents !== "number") return "—";
    var code = String(currency || "usd").toUpperCase();
    try {
      return new Intl.NumberFormat("en-US", { style: "currency", currency: code }).format(cents / 100);
    } catch (_badCode) {
      return (cents / 100).toFixed(2) + " " + code;
    }
  }

  function redirectToLogin() {
    var next = window.location.pathname + window.location.search;
    window.location.replace("/login?next=" + encodeURIComponent(next));
  }

  /** Run one section's load; 401 sends the user to login, other errors show inline. */
  async function guarded(errorId, task) {
    try {
      await task();
    } catch (err) {
      if (err.status === 401) {
        redirectToLogin();
        return;
      }
      $(errorId).textContent = err.message;
    }
  }

  // ---------- Account & usage ----------

  function renderMeter(key, used, limit) {
    var meter = $("usage-" + key);
    var max = Math.max(limit, 1);
    meter.max = max;
    meter.value = Math.min(used, max);
    meter.classList.toggle("is-near-limit", limit > 0 && used / limit >= NEAR_LIMIT_RATIO);
    $("usage-" + key + "-value").textContent =
      numberFormat.format(used) + " of " + numberFormat.format(limit);
  }

  function renderAccount(data) {
    var user = data.user || {};
    var usage = data.usage || {};
    var limits = data.limits || {};
    $("account-email").textContent = user.email || "—";
    var badge = $("account-plan");
    badge.textContent = planLabel(user.plan);
    badge.classList.toggle("is-pro", user.plan === "pro");
    $("account-since").textContent = formatDate(user.created_at);
    renderMeter("analyses", Number(usage.analyses_this_month) || 0, Number(limits.analyses_per_month) || 0);
    renderMeter("chat", Number(usage.chat_messages_today) || 0, Number(limits.chat_messages_per_day) || 0);
  }

  // ---------- API keys ----------

  function populateProviders(providers) {
    providerById = {};
    var options = providers.map(function (provider) {
      providerById[provider.id] = provider;
      var option = el("option", null, provider.name || provider.id);
      option.value = provider.id;
      return option;
    });
    $("key-provider").replaceChildren(...options);
    onProviderChange();
  }

  function onProviderChange() {
    var select = $("key-provider");
    var provider = providerById[select.value] || {};
    var isCustom = select.value === CUSTOM_PROVIDER;
    $("key-base-wrap").hidden = !isCustom;
    $("key-model").placeholder = provider.default_model || "";
    $("key-base-url").placeholder = provider.default_base_url || "https://api.example.com/v1";
  }

  function renderKeyRow(key) {
    var provider = providerById[key.provider];
    var name = provider ? provider.name : key.provider;
    var row = el("li", "key-row");
    var info = el("div", "key-info");
    info.appendChild(el("strong", null, name));
    info.appendChild(el("span", "mono", "Key " + (key.key_hint || "saved")));
    if (key.model) info.appendChild(el("span", "help", "Model: " + key.model));
    if (key.base_url) info.appendChild(el("span", "help", "Endpoint: " + key.base_url));

    var button = el("button", "btn btn-danger btn-sm", "Delete");
    button.type = "button";
    button.setAttribute("aria-label", "Delete " + name + " key");
    button.addEventListener("click", function () {
      deleteKey(key.provider, name, button);
    });
    row.append(info, button);
    return row;
  }

  function renderKeys(keys) {
    $("keys-list").replaceChildren(...keys.map(renderKeyRow));
    $("keys-empty").hidden = keys.length > 0;
  }

  async function loadKeys() {
    var data = await Sherlock.api("/api/keys");
    renderKeys(Array.isArray(data.keys) ? data.keys : []);
    if (!$("key-provider").options.length) {
      populateProviders(Array.isArray(data.providers) ? data.providers : []);
    }
  }

  function setKeyStatus(message, isError) {
    $("key-error").textContent = isError ? message : "";
    $("key-status").textContent = isError ? "" : message;
  }

  function buildKeyBody() {
    var provider = $("key-provider").value;
    var apiKey = $("key-api").value.trim();
    var model = $("key-model").value.trim();
    var baseUrl = $("key-base-url").value.trim();
    if (!provider) return { error: "Choose a provider." };
    if (!apiKey) return { error: "Paste your API key." };
    if (provider === CUSTOM_PROVIDER && !baseUrl) {
      return { error: "Enter the base URL for your endpoint." };
    }
    var body = { provider: provider, api_key: apiKey };
    if (model) body.model = model;
    if (provider === CUSTOM_PROVIDER) body.base_url = baseUrl;
    return { body: body };
  }

  async function onKeySubmit(event) {
    event.preventDefault();
    setKeyStatus("", false);
    var built = buildKeyBody();
    if (built.error) {
      setKeyStatus(built.error, true);
      return;
    }
    var submit = $("key-submit");
    submit.disabled = true;
    submit.setAttribute("aria-busy", "true");
    try {
      await Sherlock.api("/api/keys", { method: "POST", body: built.body });
      $("key-api").value = "";
      $("key-model").value = "";
      $("key-base-url").value = "";
      setKeyStatus((providerById[built.body.provider] || {}).name + " key saved.", false);
      await loadKeys();
    } catch (err) {
      if (err.status === 401) {
        redirectToLogin();
        return;
      }
      setKeyStatus(err.message, true);
    } finally {
      submit.disabled = false;
      submit.removeAttribute("aria-busy");
    }
  }

  async function deleteKey(provider, name, button) {
    if (!window.confirm("Remove your " + name + " key? Chat will use the platform model unless you add another key.")) {
      return;
    }
    button.disabled = true;
    try {
      await Sherlock.api("/api/keys/" + encodeURIComponent(provider), { method: "DELETE" });
      setKeyStatus("Removed " + name + " key.", false);
      await loadKeys();
    } catch (err) {
      button.disabled = false;
      if (err.status === 401) {
        redirectToLogin();
        return;
      }
      setKeyStatus(err.message, true);
    }
  }

  // ---------- Billing ----------

  function renderPayments(payments) {
    var rows = payments.map(function (payment) {
      var row = el("tr");
      row.appendChild(el("td", null, formatDate(payment.created_at)));
      row.appendChild(el("td", null, planLabel(payment.plan)));
      row.appendChild(el("td", null, formatMoney(payment.amount_cents, payment.currency)));
      row.appendChild(el("td", null, payment.status || "—"));
      return row;
    });
    $("payments-body").replaceChildren(...rows);
    $("payments-table").hidden = rows.length === 0;
    $("payments-empty").hidden = rows.length > 0;
  }

  function billingSummary(isPro, enabled) {
    if (isPro) return "You are on the Pro plan.";
    if (enabled) return "You are on the Free plan. Upgrade to Pro for higher monthly and daily limits.";
    return "You are on the Free plan. Paid plans are not available yet.";
  }

  function renderBilling(data) {
    var isPro = data.plan === "pro";
    var enabled = data.billing_enabled === true;
    var badge = $("billing-plan");
    badge.textContent = planLabel(data.plan);
    badge.classList.toggle("is-pro", isPro);
    $("billing-summary").textContent = billingSummary(isPro, enabled);

    var upgrade = $("upgrade-btn");
    upgrade.hidden = isPro;
    upgrade.disabled = !enabled;
    upgrade.textContent = enabled ? "Upgrade to Pro" : "Coming soon";
    renderPayments(Array.isArray(data.payments) ? data.payments : []);
  }

  async function onUpgrade(event) {
    var button = event.currentTarget;
    button.disabled = true;
    $("billing-error").textContent = "";
    try {
      await Sherlock.redirectToCheckout("pro");
    } catch (err) {
      button.disabled = false;
      if (err.status === 401) {
        redirectToLogin();
        return;
      }
      $("billing-error").textContent = err.message;
    }
  }

  function showCheckoutBanner() {
    var state = params.get("checkout");
    var banner = $("checkout-banner");
    if (state === "success") {
      banner.className = "banner banner-success";
      banner.textContent = "Thanks. Your payment is being confirmed, and your plan will update as soon as it is.";
      banner.hidden = false;
    } else if (state === "cancel") {
      banner.className = "banner banner-info";
      banner.textContent = "Checkout was cancelled. You are still on your current plan.";
      banner.hidden = false;
    }
  }

  function highlightUpgradeIfRequested() {
    if (params.get("upgrade") !== "pro") return;
    var panel = $("billing");
    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    panel.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    panel.classList.add("is-highlighted");
    window.setTimeout(function () {
      panel.classList.remove("is-highlighted");
    }, HIGHLIGHT_MS);
    var upgrade = $("upgrade-btn");
    if (!upgrade.hidden && !upgrade.disabled) {
      upgrade.focus({ preventScroll: true });
    }
  }

  // ---------- Init ----------

  async function init() {
    document.querySelectorAll('[data-action="logout"]').forEach(function (button) {
      button.addEventListener("click", Sherlock.logout);
    });
    showCheckoutBanner();

    var data;
    try {
      data = await Sherlock.api("/api/auth/me");
    } catch (err) {
      if (err.status === 401) {
        redirectToLogin();
        return;
      }
      $("account-error").textContent = "Could not load your account: " + err.message;
      return;
    }
    renderAccount(data);

    $("key-provider").addEventListener("change", onProviderChange);
    $("key-form").addEventListener("submit", onKeySubmit);
    $("upgrade-btn").addEventListener("click", onUpgrade);

    await guarded("key-error", loadKeys);
    await guarded("billing-error", async function () {
      renderBilling(await Sherlock.api("/api/billing/status"));
    });
    highlightUpgradeIfRequested();
  }

  init();
})();
