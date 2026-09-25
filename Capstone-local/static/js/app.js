// Shared UI behavior: sidebar collapse, modals, pill-tabs, toggles.
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

  // Toggle switches: [data-toggle] flips .on class (visual only unless data-toggle-name present -> posts to endpoint)
  document.querySelectorAll(".toggle[data-toggle]").forEach((t) => {
    t.addEventListener("click", () => t.classList.toggle("on"));
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
    label.textContent = expanded ? "Hide doses" : "Doses";
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
