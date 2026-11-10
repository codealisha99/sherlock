/* Sherlock public site — shared helpers (landing, auth, settings).
 * Plain script, no modules. Exposes window.Sherlock. Text is set via textContent only. */
(function () {
  "use strict";

  /**
   * JSON API call. Resolves with the parsed body, or throws an Error with
   * `.status`, `.retryAfter` (raw Retry-After header, if any) and `.data`.
   */
  async function api(path, options) {
    const opts = options || {};
    const method = opts.method || "GET";
    const init = {
      method: method,
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    };
    if (opts.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.body);
    }

    let res;
    try {
      res = await fetch(path, init);
    } catch (_networkError) {
      const err = new Error("Network error. Check your connection and try again.");
      err.status = 0;
      throw err;
    }

    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (_notJson) {
      data = null;
    }

    if (!res.ok) {
      const message = (data && typeof data.error === "string" && data.error)
        || "Request failed (" + res.status + ").";
      const err = new Error(message);
      err.status = res.status;
      err.retryAfter = res.headers.get("Retry-After");
      err.data = data;
      throw err;
    }
    return data;
  }

  /** Current user from GET /api/auth/me, or null when logged out. */
  async function getMe() {
    try {
      const data = await api("/api/auth/me");
      return data && data.user ? data.user : null;
    } catch (err) {
      if (err.status === 401) return null;
      throw err;
    }
  }

  /** Log out and return to the landing page. Failures are shown, not swallowed. */
  async function logout() {
    try {
      await api("/api/auth/logout", { method: "POST" });
    } catch (err) {
      window.alert("Could not log out: " + err.message);
      return;
    }
    window.location.assign("/");
  }

  /**
   * Toggle header nav items: [data-auth="guest"] shown when logged out,
   * [data-auth="user"] shown when logged in. Wires [data-action="logout"].
   * Resolves with the current user (or null).
   */
  async function initNav() {
    let me = null;
    try {
      me = await getMe();
    } catch (_err) {
      me = null; // Treat an unreachable API as logged out; nav stays usable.
    }
    document.querySelectorAll('[data-auth="guest"]').forEach(function (node) {
      node.hidden = Boolean(me);
    });
    document.querySelectorAll('[data-auth="user"]').forEach(function (node) {
      node.hidden = !me;
    });
    document.querySelectorAll('[data-action="logout"]').forEach(function (btn) {
      btn.addEventListener("click", logout);
    });
    return me;
  }

  /** Only https checkout URLs are followed; anything else is rejected. */
  function checkoutUrlOrThrow(raw) {
    const url = new URL(String(raw || ""));
    if (url.protocol !== "https:") {
      throw new Error("Unexpected checkout address from the server.");
    }
    return url.href;
  }

  /** Create a Stripe Checkout session for `plan` and navigate to it. */
  async function redirectToCheckout(plan) {
    const data = await api("/api/billing/checkout", {
      method: "POST",
      body: { plan: plan },
    });
    window.location.assign(checkoutUrlOrThrow(data && data.checkout_url));
  }

  /** Small DOM builder that always sets text via textContent. */
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  /* ---------- Page chrome: header rule, mobile menu, scroll reveals ---------- */

  const SCROLL_RULE_PX = 4;

  /** The header's bottom rule appears once the page has scrolled (passive listener). */
  function initHeaderRule(header) {
    function update() {
      header.classList.toggle("is-scrolled", window.scrollY > SCROLL_RULE_PX);
    }
    update();
    window.addEventListener("scroll", update, { passive: true });
  }

  /** Disclosure menu for small screens: aria-expanded on the button; Escape or a link closes it. */
  function initMenuToggle(header) {
    const button = header.querySelector(".nav-toggle");
    const panel = button && document.getElementById(button.getAttribute("aria-controls") || "");
    if (!button || !panel) return;

    function setOpen(open) {
      header.classList.toggle("is-open", open);
      button.setAttribute("aria-expanded", String(open));
    }

    button.addEventListener("click", function () {
      setOpen(button.getAttribute("aria-expanded") !== "true");
    });
    panel.addEventListener("click", function (event) {
      if (event.target.closest && event.target.closest("a")) setOpen(false);
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && header.classList.contains("is-open")) {
        setOpen(false);
        button.focus();
      }
    });
  }

  /** Reveal [data-reveal] elements once, when they first enter the viewport. */
  function initReveal() {
    const nodes = document.querySelectorAll("[data-reveal]");
    if (!("IntersectionObserver" in window)) {
      nodes.forEach(function (node) { node.classList.add("is-revealed"); });
      return;
    }
    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-revealed");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.15, rootMargin: "0px 0px -6% 0px" });
    nodes.forEach(function (node) { observer.observe(node); });
  }

  /** Adds html.js so CSS can hide reveal targets only when scripts will reveal them. */
  function initPageChrome() {
    document.documentElement.classList.add("js");
    const header = document.querySelector(".site-header");
    if (header) {
      initHeaderRule(header);
      initMenuToggle(header);
    }
    initReveal();
  }

  initPageChrome();

  window.Sherlock = {
    api: api,
    getMe: getMe,
    logout: logout,
    initNav: initNav,
    redirectToCheckout: redirectToCheckout,
    el: el,
  };
})();
