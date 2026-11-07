// Shared UI behavior: sidebar collapse, modals, pill-tabs.
(function () {
  const sidebar = document.getElementById("sidebar");
  const toggleBtn = document.getElementById("sidebarToggle");
  if (toggleBtn && sidebar) {
    const KEY = "iv_sidebar_collapsed";

    // The arrow direction is handled in CSS (.sidebar.collapsed rotates the
    // chevron), so this only has to manage state and the accessible labels.
    function apply(collapsed) {
      sidebar.classList.toggle("collapsed", collapsed);
      const label = collapsed ? "Expand sidebar" : "Collapse sidebar";
      toggleBtn.setAttribute("title", label);
      toggleBtn.setAttribute("aria-label", label);
      toggleBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    }

    apply(localStorage.getItem(KEY) === "1");

    toggleBtn.addEventListener("click", () => {
      const collapsed = !sidebar.classList.contains("collapsed");
      apply(collapsed);
      localStorage.setItem(KEY, collapsed ? "1" : "0");
      hideSidebarTooltip();
    });
  }

  // Mobile nav: below 860px the sidebar is a slide-in drawer (see .sidebar in
  // style.css), opened by the topbar's hamburger button and closed by its own
  // backdrop. Independent of the desktop collapse toggle above - "collapsed"
  // (icons-only) and "mobile-open" (full drawer over content) are different
  // states that never apply at the same time, since the breakpoints don't
  // overlap.
  const menuToggle = document.getElementById("menuToggle");
  const backdrop = document.getElementById("sidebarBackdrop");
  if (menuToggle && sidebar && backdrop) {
    function setMobileOpen(open) {
      sidebar.classList.toggle("mobile-open", open);
      backdrop.classList.toggle("show", open);
      menuToggle.setAttribute("aria-expanded", open ? "true" : "false");
      menuToggle.setAttribute("aria-label", open ? "Close navigation menu" : "Open navigation menu");
      document.body.classList.toggle("no-scroll", open);
    }
    menuToggle.addEventListener("click", () => setMobileOpen(!sidebar.classList.contains("mobile-open")));
    backdrop.addEventListener("click", () => setMobileOpen(false));
  }

  // Collapsed-sidebar tooltips: a nav icon alone doesn't say what it does, and
  // hovering the rail used to show nothing. A floating label fixes that
  // without the whole-page reflow a hover-to-expand sidebar would cause.
  // position:fixed (not absolute) is deliberate - .sidebar-nav scrolls its own
  // content and clips anything absolutely positioned past its right edge, so
  // only a viewport-fixed element can sit outside the rail.
  let sidebarTip = null;
  function hideSidebarTooltip() {
    if (sidebarTip) sidebarTip.classList.remove("show");
  }
  if (sidebar) {
    sidebar.querySelectorAll(".sidebar-nav .nav-item").forEach((item) => {
      item.addEventListener("mouseenter", () => {
        if (!sidebar.classList.contains("collapsed")) return;
        const label = item.querySelector(".label");
        if (!label) return;
        if (!sidebarTip) {
          sidebarTip = document.createElement("div");
          sidebarTip.className = "sidebar-tooltip";
          document.body.appendChild(sidebarTip);
        }
        sidebarTip.textContent = label.textContent;
        const r = item.getBoundingClientRect();
        sidebarTip.style.top = r.top + r.height / 2 + "px";
        sidebarTip.style.left = r.right + 10 + "px";
        sidebarTip.classList.add("show");
      });
      item.addEventListener("mouseleave", hideSidebarTooltip);
    });
  }

  // Modals: any [data-open-modal="modalId"] opens #modalId; any [data-close-modal] closes nearest .modal-scrim
  document.querySelectorAll("[data-open-modal]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-open-modal");
      const modal = document.getElementById(id);
      if (modal) modal.classList.add("open");
    });
  });
  document.querySelectorAll("[data-close-modal]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const scrim = btn.closest(".modal-scrim");
      if (scrim) scrim.classList.remove("open");
    });
  });
  document.querySelectorAll(".modal-scrim").forEach((scrim) => {
    scrim.addEventListener("click", (e) => {
      if (e.target === scrim) scrim.classList.remove("open");
    });
  });

  // Pill tabs: [data-tabgroup] wraps [data-tab] buttons; [data-tabpanel] panels toggle by matching value
  document.querySelectorAll("[data-tabgroup]").forEach((group) => {
    const groupName = group.getAttribute("data-tabgroup");
    function activate(val) {
      const btn = group.querySelector(`[data-tab="${val}"]`);
      if (!btn) return false;
      group.querySelectorAll("[data-tab]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(`[data-tabpanel="${groupName}"]`).forEach((panel) => {
        panel.style.display = panel.getAttribute("data-tabvalue") === val ? "" : "none";
      });
      return true;
    }

    group.querySelectorAll("[data-tab]").forEach((btn) => {
      btn.addEventListener("click", () => activate(btn.getAttribute("data-tab")));
    });

    // Open a specific tab from the URL, e.g. /rhu/risk#atrisk. Lets a link
    // point at the tab a user actually wants rather than the page default.
    if (window.location.hash) {
      const val = window.location.hash.slice(1);
      if (activate(val)) {
        // The panel was display:none when the page loaded, so the browser could
        // not scroll to it natively. Scroll once it is visible.
        const panel = document.querySelector(
          `[data-tabpanel="${groupName}"][data-tabvalue="${val}"]`);
        if (panel) {
          requestAnimationFrame(() =>
            panel.scrollIntoView({ behavior: "smooth", block: "start" }));
        }
      }
    }
  });

  // Generic confirm-on-click for destructive actions
  document.querySelectorAll("[data-confirm]").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (!confirm(el.getAttribute("data-confirm"))) {
        e.preventDefault();
        e.stopPropagation();
      }
    });
  });
})();

/* ---------------------------------------------------------------------------
   Account menu (topbar). Closes on outside click, on Escape, and returns focus
   to the trigger so keyboard users are not stranded inside a closed menu.
--------------------------------------------------------------------------- */
(function () {
  var btn = document.getElementById("accountBtn");
  var menu = document.getElementById("accountDropdown");
  if (!btn || !menu) return;

  function open() {
    menu.hidden = false;
    btn.setAttribute("aria-expanded", "true");
    var first = menu.querySelector("[role=menuitem]");
    if (first) first.focus();
  }
  function close(refocus) {
    menu.hidden = true;
    btn.setAttribute("aria-expanded", "false");
    if (refocus) btn.focus();
  }

  btn.addEventListener("click", function (e) {
    e.stopPropagation();
    if (menu.hidden) { open(); } else { close(false); }
  });

  document.addEventListener("click", function (e) {
    if (!menu.hidden && !menu.contains(e.target) && e.target !== btn) close(false);
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !menu.hidden) close(true);
  });

  // Arrow keys move between items once the menu is open.
  menu.addEventListener("keydown", function (e) {
    var items = Array.prototype.slice.call(menu.querySelectorAll("[role=menuitem]"));
    var i = items.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length].focus(); }
    if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
  });
})();

/* ---------------------------------------------------------------------------
   Child Records: collapse the 15 dose columns.
   They sit between the identity columns and Actions, so with them shown the
   action buttons are off screen and need a sideways scroll. Collapsed by
   default; the choice is remembered.
--------------------------------------------------------------------------- */
(function () {
  var btn = document.getElementById("doseToggle");
  if (!btn) return;
  var label = document.getElementById("doseToggleLabel");
  var table = btn.closest("table");
  var KEY = "iv_doses_expanded";

  function apply(expanded) {
    table.classList.toggle("doses-expanded", expanded);
    btn.setAttribute("aria-expanded", expanded ? "true" : "false");
    label.textContent = expanded ? "Hide doses" : "Expand doses";
    btn.title = expanded ? "Hide the individual dose columns"
                         : "Show the individual dose columns";
  }

  apply(localStorage.getItem(KEY) === "1");

  btn.addEventListener("click", function () {
    var expanded = !table.classList.contains("doses-expanded");
    apply(expanded);
    localStorage.setItem(KEY, expanded ? "1" : "0");
  });
})();

/* ---------------------------------------------------------------------------
   Custom select: wraps every select.form-control in a styled trigger+panel
   (see .select-panel in style.css) so the open dropdown matches the rest of
   the app instead of the browser's unstyleable native option list. The real
   <select> stays in the DOM, visually hidden, as the actual form field - a
   change here sets its value and dispatches a real 'change' event, so every
   onchange="..." and addEventListener('change', ...) a template already has
   keeps working unmodified. If this script fails to load, the select is
   simply left as a normal (still chevron-fixed, see style.css) native one.
--------------------------------------------------------------------------- */
(function () {
  var CHEVRON_SVG = '<svg class="st-chevron" width="13" height="13" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<polyline points="6 9 12 15 18 9"></polyline></svg>';
  var uid = 0;

  function closeAllPanels(except) {
    document.querySelectorAll(".select-panel").forEach(function (panel) {
      if (panel === except || panel.hidden) return;
      panel.hidden = true;
      var shell = panel.parentElement;
      var trigger = shell.querySelector(".select-trigger") || shell;
      trigger.setAttribute("aria-expanded", "false");
      trigger.removeAttribute("aria-activedescendant");
    });
  }

  function enhance(select) {
    if (select.dataset.customized) return;
    select.dataset.customized = "1";
    uid += 1;

    var shell = document.createElement("div");
    shell.className = "select-shell";
    if (select.getAttribute("style")) shell.setAttribute("style", select.getAttribute("style"));
    select.parentNode.insertBefore(shell, select);
    shell.appendChild(select);

    var trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = select.className + " select-trigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    if (select.disabled) trigger.disabled = true;
    if (select.id) trigger.setAttribute("aria-labelledby", select.id);

    var valueSpan = document.createElement("span");
    valueSpan.className = "st-value";
    trigger.appendChild(valueSpan);
    trigger.insertAdjacentHTML("beforeend", CHEVRON_SVG);
    shell.appendChild(trigger);

    attachBehavior(select, shell, trigger, valueSpan, trigger);
  }

  // Same dropdown, for a <select> already sitting inside the app's other
  // select-styled wrapper (.filter-select: a leading icon + the select's own
  // text + a static chevron). Reuses that wrapper as the shell instead of
  // building a new one, so the icon stays and only the native option list
  // is swapped out - the select's own text layer becomes the value display.
  function enhanceFilterSelect(select) {
    if (select.dataset.customized) return;
    select.dataset.customized = "1";
    uid += 1;

    var shell = select.closest(".filter-select");
    if (!shell) return;
    shell.classList.add("select-shell", "fs-shell");

    var valueSpan = document.createElement("span");
    valueSpan.className = "st-value fs-value";
    valueSpan.setAttribute("aria-hidden", "true");
    shell.insertBefore(valueSpan, select);

    shell.setAttribute("tabindex", "0");
    shell.setAttribute("role", "button");
    shell.setAttribute("aria-haspopup", "listbox");
    shell.setAttribute("aria-expanded", "false");
    if (select.id) shell.setAttribute("aria-labelledby", select.id);

    attachBehavior(select, shell, shell, valueSpan, shell);
  }

  // Shared panel + keyboard/pointer wiring for both enhance() and
  // enhanceFilterSelect(). `clickTarget` opens/closes the panel;
  // `expandedTarget` carries aria-expanded/aria-activedescendant (the
  // trigger button for a plain select, the whole shell for a filter-select,
  // since that variant has no separate trigger element of its own).
  function attachBehavior(select, shell, clickTarget, valueSpan, expandedTarget) {
    var panel = document.createElement("div");
    panel.className = "select-panel";
    panel.setAttribute("role", "listbox");
    panel.hidden = true;
    shell.appendChild(panel);

    var options = [];
    function buildOptions() {
      panel.innerHTML = "";
      options = Array.prototype.map.call(select.options, function (opt, i) {
        // <optgroup>: a non-selectable heading before the group's first option.
        var grp = opt.parentElement;
        if (grp && grp.tagName === "OPTGROUP" && grp.firstElementChild === opt) {
          var head = document.createElement("div");
          head.className = "select-group";
          head.setAttribute("role", "presentation");
          head.textContent = grp.label;
          panel.appendChild(head);
        }
        var row = document.createElement("div");
        row.className = "select-option";
        row.id = "select-opt-" + uid + "-" + i;
        row.setAttribute("role", "option");
        row.title = opt.textContent;
        var label = document.createElement("span");
        label.className = "so-label";
        label.textContent = opt.textContent;
        row.appendChild(label);
        if (opt.disabled) row.setAttribute("aria-disabled", "true");
        row.dataset.value = opt.value;
        panel.appendChild(row);
        return row;
      });
      syncFromSelect();
    }

    function syncFromSelect() {
      var selectedOpt = select.options[select.selectedIndex];
      valueSpan.textContent = selectedOpt ? selectedOpt.textContent : "";
      options.forEach(function (row) {
        row.classList.toggle("is-selected", row.dataset.value === select.value);
      });
    }

    function setActive(row) {
      options.forEach(function (r) { r.classList.remove("is-active"); });
      if (!row) return;
      row.classList.add("is-active");
      expandedTarget.setAttribute("aria-activedescendant", row.id);
      row.scrollIntoView({ block: "nearest" });
    }

    function openPanel() {
      closeAllPanels(panel);
      panel.hidden = false;
      expandedTarget.setAttribute("aria-expanded", "true");
      setActive(panel.querySelector(".is-selected") || options[0]);
    }
    function closePanel(refocus) {
      panel.hidden = true;
      expandedTarget.setAttribute("aria-expanded", "false");
      expandedTarget.removeAttribute("aria-activedescendant");
      if (refocus) clickTarget.focus();
    }
    function choose(row) {
      if (!row || row.getAttribute("aria-disabled") === "true") return;
      select.value = row.dataset.value;
      syncFromSelect();
      select.dispatchEvent(new Event("change", { bubbles: true }));
      closePanel(true);
    }

    clickTarget.addEventListener("click", function () {
      if (panel.hidden) openPanel(); else closePanel(false);
    });
    clickTarget.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (panel.hidden) { openPanel(); return; }
        var i = options.indexOf(panel.querySelector(".is-active"));
        var next = e.key === "ArrowDown" ? Math.min(i + 1, options.length - 1) : Math.max(i - 1, 0);
        setActive(options[next]);
      } else if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (panel.hidden) openPanel(); else choose(panel.querySelector(".is-active"));
      } else if (e.key === "Escape" && !panel.hidden) {
        e.preventDefault();
        closePanel(true);
      } else if (e.key === "Tab") {
        closePanel(false);
      }
    });
    panel.addEventListener("mousemove", function (e) {
      var row = e.target.closest(".select-option");
      if (row) setActive(row);
    });
    panel.addEventListener("click", function (e) {
      // The filter-select variant uses the whole shell (which contains this
      // panel) as its click target, so without this a click on a row would
      // bubble up and immediately reopen the panel it just closed.
      e.stopPropagation();
      var row = e.target.closest(".select-option");
      if (row) choose(row);
    });

    // Keeps the trigger's label correct if something else (e.g. a reset
    // button, or a filter script elsewhere on the page) changes the
    // underlying select's value directly.
    select.addEventListener("change", syncFromSelect);

    buildOptions();
  }

  document.querySelectorAll("select.form-control").forEach(enhance);
  document.querySelectorAll(".filter-select select").forEach(enhanceFilterSelect);

  document.addEventListener("click", function (e) {
    if (!e.target.closest(".select-shell")) closeAllPanels(null);
  });
})();

// Printing: charts are drawn for the screen, and Chrome lays out the printed
// page without telling Chart.js, so a chart kept its screen size and spilled
// over its neighbours. Redraw every chart when print layout starts and ends.
(function () {
  function resizeCharts() {
    if (!window.Chart || !Chart.instances) return;
    Object.keys(Chart.instances).forEach(function (id) { Chart.instances[id].resize(); });
  }
  if (window.matchMedia) {
    var mq = window.matchMedia("print");
    if (mq.addEventListener) mq.addEventListener("change", resizeCharts);
    else if (mq.addListener) mq.addListener(resizeCharts);
  }
  window.addEventListener("beforeprint", resizeCharts);
  window.addEventListener("afterprint", resizeCharts);
})();

// Field checks for the child forms. An input opts in with data-validate:
//   name       letters, spaces, . ' - only (data-required for a required one)
//   phone      Philippine mobile, 09XXXXXXXXX or +639XXXXXXXXX
//   dob        not in the future
//   registered not in the future, not before the form's date of birth
// The message appears under the field once it has been left, updates as the
// user types, and blocks submitting. app.py checks the same rules.
(function () {
  var NAME_OK = /^[\p{L} .'\-]+$/u;
  var HAS_LETTER = /\p{L}/u;
  function todayIso() {
    var d = new Date();
    return [d.getFullYear(), String(d.getMonth() + 1).padStart(2, "0"), String(d.getDate()).padStart(2, "0")].join("-");
  }
  function message(input) {
    var v = (input.value || "").trim();
    var kind = input.getAttribute("data-validate");
    var label = input.getAttribute("data-label") || "This field";
    if (!v) return input.hasAttribute("data-required") ? label + " is required." : "";
    // Edit forms: a value that came from the record and was not touched is
    // left alone (imported records may use an older format).
    if (input.hasAttribute("data-only-if-changed") && input.value === input.defaultValue) return "";
    if (kind === "name" && (!NAME_OK.test(v) || !HAS_LETTER.test(v)))
      return label + " can only contain letters, spaces, periods, apostrophes and hyphens.";
    if (kind === "phone" && !/^(09\d{9}|\+639\d{9})$/.test(v.replace(/[\s-]/g, "")))
      return "Enter an 11-digit mobile number, e.g. 09171234567.";
    if (kind === "dob" && v > todayIso()) return "Date of birth cannot be in the future.";
    if (kind === "registered") {
      if (v > todayIso()) return "Date of registration cannot be in the future.";
      var dob = input.form && input.form.querySelector('[data-validate="dob"]');
      if (dob && dob.value && v < dob.value) return "Date of registration cannot be before the date of birth.";
    }
    return "";
  }
  function show(input) {
    var msg = message(input);
    var box = input.parentElement.querySelector(".field-error[data-for='" + input.name + "']");
    if (!box) {
      box = document.createElement("div");
      box.className = "field-error";
      box.setAttribute("data-for", input.name);
      box.setAttribute("role", "alert");
      input.insertAdjacentElement("afterend", box);
    }
    box.textContent = msg;
    box.hidden = !msg;
    input.classList.toggle("is-invalid", !!msg);
    input.setAttribute("aria-invalid", msg ? "true" : "false");
    return !msg;
  }
  window.ivValidateForm = function (form) {
    var ok = true, first = null;
    form.querySelectorAll("[data-validate]").forEach(function (input) {
      input.dataset.touched = "1";
      if (!show(input)) { ok = false; first = first || input; }
    });
    if (first) first.focus();
    return ok;
  };
  window.ivClearErrors = function (form) {
    form.querySelectorAll("[data-validate]").forEach(function (input) {
      delete input.dataset.touched;
      input.classList.remove("is-invalid");
      input.removeAttribute("aria-invalid");
    });
    form.querySelectorAll(".field-error").forEach(function (box) { box.hidden = true; box.textContent = ""; });
  };
  document.querySelectorAll("form").forEach(function (form) {
    var inputs = form.querySelectorAll("[data-validate]");
    if (!inputs.length) return;
    inputs.forEach(function (input) {
      input.addEventListener("blur", function () { input.dataset.touched = "1"; show(input); });
      input.addEventListener("input", function () { if (input.dataset.touched) show(input); });
      input.addEventListener("change", function () {
        input.dataset.touched = "1"; show(input);
        // A new date of birth can make the registration date valid or not.
        if (input.getAttribute("data-validate") === "dob") {
          var reg = form.querySelector('[data-validate="registered"]');
          if (reg && reg.dataset.touched) show(reg);
        }
      });
    });
    form.addEventListener("submit", function (e) {
      if (!window.ivValidateForm(form)) e.preventDefault();
    });
  });
})();

// Register Child: closing the form clears it, so the next "Add Child" starts
// blank. If something was already typed or ticked, ask before discarding.
(function () {
  // A form counts as changed when any field differs from its state at page
  // load (comparing against defaultSelected would flag a select whose first
  // option is only selected implicitly).
  var initial = new WeakMap();
  function snapshot(form) {
    return Array.prototype.map.call(form.elements, function (el) {
      return el.type === "checkbox" || el.type === "radio" ? el.checked : el.value;
    });
  }
  function isDirty(form) {
    var before = initial.get(form);
    if (!before) return false;
    var now = snapshot(form);
    return now.some(function (v, i) {
      var el = form.elements[i];
      if (!el.name || el.type === "hidden" || el.type === "submit" || el.type === "button") return false;
      return v !== before[i];
    });
  }
  function resetForm(form) {
    form.reset();
    if (window.ivClearErrors) window.ivClearErrors(form);
    // Let the form's own scripts react: hide vaccine date rows, recompute the
    // BHW eligibility hints, refresh the custom dropdown labels.
    form.querySelectorAll("input[type=checkbox], select, input[type=date]").forEach(function (el) {
      el.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }
  function confirmDiscard(onYes) {
    var dlg = document.getElementById("ivDiscardDialog");
    if (!dlg) {
      dlg = document.createElement("div");
      dlg.id = "ivDiscardDialog";
      dlg.className = "modal-scrim discard-scrim";
      dlg.innerHTML =
        '<div class="modal-box discard-box" role="alertdialog" aria-modal="true" aria-labelledby="ivDiscardTitle">' +
        '<div class="modal-head"><h3 id="ivDiscardTitle">Discard this child\'s details?</h3></div>' +
        '<div class="modal-body"><p style="margin:0;font-size:13.5px;color:var(--text-secondary);line-height:1.6;">' +
        "What you typed in the Register New Child form will be cleared.</p></div>" +
        '<div class="modal-footer"><button type="button" class="btn btn-outline" data-discard="no">Keep editing</button>' +
        '<button type="button" class="btn btn-danger" data-discard="yes">Discard</button></div></div>';
      document.body.appendChild(dlg);
    }
    dlg.classList.add("open");
    var yes = dlg.querySelector('[data-discard="yes"]'), no = dlg.querySelector('[data-discard="no"]');
    function done(result) {
      dlg.classList.remove("open");
      yes.onclick = no.onclick = dlg.onclick = null;
      if (result) onYes();
    }
    yes.onclick = function () { done(true); };
    no.onclick = function () { done(false); };
    dlg.onclick = function (e) { if (e.target === dlg) done(false); };
    no.focus();
  }
  // Runs before the generic close handlers (capture phase) so it can keep the
  // modal open when the user chooses "Keep editing".
  function guardClose(scrim, e) {
    var form = scrim.querySelector("form");
    if (!form || !scrim.hasAttribute("data-reset-on-close")) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    if (!isDirty(form)) { resetForm(form); scrim.classList.remove("open"); return; }
    confirmDiscard(function () { resetForm(form); scrim.classList.remove("open"); });
  }
  document.querySelectorAll(".modal-scrim[data-reset-on-close]").forEach(function (scrim) {
    var f = scrim.querySelector("form");
    if (f) initial.set(f, snapshot(f));
    scrim.querySelectorAll("[data-close-modal]").forEach(function (btn) {
      btn.addEventListener("click", function (e) { guardClose(scrim, e); }, true);
    });
    scrim.addEventListener("click", function (e) { if (e.target === scrim) guardClose(scrim, e); }, true);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    var discard = document.getElementById("ivDiscardDialog");
    if (discard && discard.classList.contains("open")) { discard.querySelector('[data-discard="no"]').click(); return; }
    var open = document.querySelector(".modal-scrim.open[data-reset-on-close]");
    if (open) guardClose(open, e);
  });
})();
