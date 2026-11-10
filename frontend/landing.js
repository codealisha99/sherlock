/* Landing page: repo URL form, live pricing cards, checkout CTA. */
(function () {
  "use strict";

  var Sherlock = window.Sherlock;
  var REPO_RE = /^https:\/\/github\.com\/[A-Za-z0-9_.-]{1,39}\/[A-Za-z0-9_.-]{1,100}$/;
  var ANALYZE_PATH = "/app";
  var PRO_NEXT = "/settings?upgrade=pro";
  var PLAN_FEATURED = "pro";

  var form = document.getElementById("repo-form");
  var input = document.getElementById("repo-url");
  var errorEl = document.getElementById("repo-error");
  var grid = document.getElementById("pricing-grid");
  var pricingStatus = document.getElementById("pricing-status");
  var currentUser = null;

  var numberFormat = new Intl.NumberFormat("en-US");

  function setRepoError(message) {
    errorEl.textContent = message;
    input.setAttribute("aria-invalid", message ? "true" : "false");
  }

  function normalizeRepoUrl(raw) {
    return String(raw || "").trim().replace(/\/+$/, "");
  }

  function onSubmit(event) {
    event.preventDefault();
    var url = normalizeRepoUrl(input.value);
    if (!REPO_RE.test(url)) {
      setRepoError("Enter a GitHub repository URL like https://github.com/owner/repo.");
      input.focus();
      return;
    }
    setRepoError("");
    var target = ANALYZE_PATH + "?repo=" + encodeURIComponent(url);
    window.location.assign(
      currentUser ? target : "/signup?next=" + encodeURIComponent(target)
    );
  }

  function formatPrice(cents) {
    if (!cents) return "$0";
    return "$" + numberFormat.format(Math.round(cents / 100));
  }

  function planFeatures(plan) {
    return [
      numberFormat.format(plan.analyses_per_month) + " repository analyses per month",
      numberFormat.format(plan.chat_messages_per_day) + " chat messages per day",
      "Repositories up to " + numberFormat.format(plan.max_repo_mb) + " MB",
    ];
  }

  function buildAction(plan, billingEnabled) {
    if (plan.id !== PLAN_FEATURED) {
      var link = Sherlock.el("a", "btn btn-secondary btn-block", "Get started free");
      link.href = "/signup";
      return link;
    }
    var button = Sherlock.el("button", "btn btn-primary btn-block", "Upgrade to Pro");
    button.type = "button";
    if (!billingEnabled) {
      button.textContent = "Coming soon";
      button.disabled = true;
      return button;
    }
    button.addEventListener("click", function () {
      startProCheckout(button);
    });
    return button;
  }

  function startProCheckout(button) {
    if (!currentUser) {
      window.location.assign("/signup?next=" + encodeURIComponent(PRO_NEXT));
      return;
    }
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    pricingStatus.textContent = "";
    Sherlock.redirectToCheckout(PLAN_FEATURED).catch(function (err) {
      button.disabled = false;
      button.removeAttribute("aria-busy");
      pricingStatus.textContent = "Could not start checkout: " + err.message;
    });
  }

  function renderPlan(plan, billingEnabled) {
    var isFeatured = plan.id === PLAN_FEATURED;
    var card = Sherlock.el("article", "plan-card" + (isFeatured ? " is-featured" : ""));

    var head = Sherlock.el("div", "plan-name");
    head.appendChild(Sherlock.el("h3", null, plan.name));
    if (isFeatured) head.appendChild(Sherlock.el("span", "plan-pill", "Most analysis"));
    card.appendChild(head);

    var price = Sherlock.el("p", "price");
    price.appendChild(Sherlock.el("span", "price-amount", formatPrice(plan.price_cents)));
    price.appendChild(Sherlock.el("span", "price-period", plan.price_cents ? "per month" : "forever"));
    card.appendChild(price);

    var list = Sherlock.el("ul", "plan-features");
    planFeatures(plan).forEach(function (text) {
      list.appendChild(Sherlock.el("li", null, text));
    });
    card.appendChild(list);

    card.appendChild(buildAction(plan, billingEnabled));
    return card;
  }

  async function loadPlans() {
    try {
      var data = await Sherlock.api("/api/billing/plans");
      var plans = Array.isArray(data && data.plans) ? data.plans : [];
      var billingEnabled = Boolean(data && data.billing_enabled);
      grid.replaceChildren(...plans.map(function (plan) {
        return renderPlan(plan, billingEnabled);
      }));
      pricingStatus.textContent = plans.length ? "" : "No plans are published yet.";
    } catch (err) {
      grid.replaceChildren();
      pricingStatus.textContent = "Pricing could not be loaded (" + err.message + "). Try again shortly.";
    }
  }

  form.addEventListener("submit", onSubmit);
  input.addEventListener("input", function () {
    if (errorEl.textContent) setRepoError("");
  });

  Sherlock.initNav().then(function (me) {
    currentUser = me;
  });
  loadPlans();
})();
