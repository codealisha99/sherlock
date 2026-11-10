/* Login and signup (one page, mode from location.pathname). */
(function () {
  "use strict";

  var Sherlock = window.Sherlock;
  var EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  var DEFAULT_REDIRECT = "/app";
  var MAX_NEXT_LENGTH = 2048;

  var mode = window.location.pathname.replace(/\/+$/, "") === "/signup" ? "signup" : "login";
  var params = new URLSearchParams(window.location.search);

  var form = document.getElementById("auth-form");
  var emailInput = document.getElementById("auth-email");
  var passwordInput = document.getElementById("auth-password");
  var rulesList = document.getElementById("password-rules");
  var errorEl = document.getElementById("auth-error");
  var submitBtn = document.getElementById("auth-submit");
  var titleEl = document.getElementById("auth-title");
  var subtitleEl = document.getElementById("auth-subtitle");
  var switchText = document.getElementById("auth-switch-text");
  var switchLink = document.getElementById("auth-switch-link");

  /**
   * Open-redirect guard: only same-origin relative paths are followed.
   * Must start with a single "/", and must not be "//" or "/\" (browsers
   * treat backslashes as slashes), and must not contain control characters.
   */
  function safeNext(raw) {
    if (typeof raw !== "string" || raw.length === 0 || raw.length > MAX_NEXT_LENGTH) {
      return DEFAULT_REDIRECT;
    }
    if (!raw.startsWith("/") || raw.startsWith("//") || raw.startsWith("/\\")) {
      return DEFAULT_REDIRECT;
    }
    if (/[\u0000-\u001f\u007f\\]/.test(raw)) {
      return DEFAULT_REDIRECT;
    }
    try {
      var url = new URL(raw, window.location.origin);
      if (url.origin !== window.location.origin) return DEFAULT_REDIRECT;
      return url.pathname + url.search + url.hash;
    } catch (_invalid) {
      return DEFAULT_REDIRECT;
    }
  }

  var nextPath = safeNext(params.get("next"));

  function passwordChecks(value) {
    return {
      length: value.length >= 10,
      letter: /[A-Za-z]/.test(value),
      digit: /\d/.test(value),
    };
  }

  function updateRules() {
    var checks = passwordChecks(passwordInput.value);
    rulesList.querySelectorAll("li[data-rule]").forEach(function (item) {
      item.classList.toggle("is-met", Boolean(checks[item.dataset.rule]));
    });
    return checks.length && checks.letter && checks.digit;
  }

  function setMode(nextMode) {
    var isSignup = nextMode === "signup";
    mode = nextMode;
    document.title = (isSignup ? "Sign up" : "Log in") + " — Sherlock";
    titleEl.textContent = isSignup ? "Create your account" : "Log in";
    subtitleEl.textContent = isSignup
      ? "Analyze repositories, explore the graph, and chat with any node."
      : "Welcome back. Pick up where you left off.";
    submitBtn.textContent = isSignup ? "Create account" : "Log in";
    passwordInput.setAttribute("autocomplete", isSignup ? "new-password" : "current-password");
    rulesList.hidden = !isSignup;
    switchText.textContent = isSignup ? "Already have an account?" : "New to Sherlock?";
    switchLink.textContent = isSignup ? "Log in" : "Create an account";
    var path = isSignup ? "/login" : "/signup";
    switchLink.href = path + (params.has("next") ? "?next=" + encodeURIComponent(nextPath) : "");
  }

  function setError(message) {
    errorEl.textContent = message;
  }

  function retryMessage(retryAfter) {
    var seconds = parseInt(retryAfter, 10);
    if (Number.isFinite(seconds) && seconds > 0) {
      return "Too many attempts. Try again in " + seconds + " seconds.";
    }
    return "Too many attempts. Try again shortly.";
  }

  /** Map API status codes to user-facing messages. */
  function messageFor(err) {
    switch (err.status) {
      case 409:
        return "An account with this email already exists. Try logging in instead.";
      case 401:
        return "Invalid email or password.";
      case 423:
        return "This account is temporarily locked after repeated failed attempts. Try again later.";
      case 429:
        return retryMessage(err.retryAfter);
      default:
        return err.message;
    }
  }

  function validate() {
    var email = emailInput.value.trim();
    if (!EMAIL_RE.test(email)) {
      return "Enter a valid email address.";
    }
    if (mode === "signup" && !updateRules()) {
      return "Choose a password with at least 10 characters, including a letter and a number.";
    }
    if (!passwordInput.value) {
      return "Enter your password.";
    }
    return "";
  }

  function setBusy(busy) {
    submitBtn.disabled = busy;
    if (busy) {
      submitBtn.setAttribute("aria-busy", "true");
    } else {
      submitBtn.removeAttribute("aria-busy");
    }
  }

  async function onSubmit(event) {
    event.preventDefault();
    setError("");
    var problem = validate();
    if (problem) {
      setError(problem);
      return;
    }

    setBusy(true);
    var endpoint = mode === "signup" ? "/api/auth/signup" : "/api/auth/login";
    try {
      await Sherlock.api(endpoint, {
        method: "POST",
        body: { email: emailInput.value.trim(), password: passwordInput.value },
      });
      window.location.replace(nextPath);
    } catch (err) {
      passwordInput.value = "";
      updateRules();
      setError(messageFor(err));
      setBusy(false);
      passwordInput.focus();
    }
  }

  async function redirectIfLoggedIn() {
    try {
      var me = await Sherlock.getMe();
      if (me) window.location.replace(nextPath);
    } catch (_unreachable) {
      // API unreachable: keep the form usable; submit will report the error.
    }
  }

  setMode(mode);
  form.addEventListener("submit", onSubmit);
  passwordInput.addEventListener("input", updateRules);
  redirectIfLoggedIn();
})();
