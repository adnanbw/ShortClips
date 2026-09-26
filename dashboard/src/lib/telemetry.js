// Lightweight custom-event helper (OpenPanel).
//
// The hosted openshorts.app build loads OpenPanel (see index.html), which
// exposes `window.op`. Self-hosted builds, ad-blockers or offline dev simply
// won't have it — every call here is a safe no-op in that case, so analytics
// can never break the app or leak into the open-source experience.
//
// THE FILENAME IS PART OF THAT GUARANTEE. This module was `analytics.js`, and
// uBlock Origin and friends block any request whose path matches `analytics`
// — including this one. The no-op above only covers a MISSING `window.op`;
// it cannot help when the module itself never arrives, because a failed ES
// import aborts the whole bundle and React never mounts. Self-hosted, that is
// a blank page with one line in the console:
//
//   GET .../src/lib/analytics.js  net::ERR_BLOCKED_BY_CLIENT
//
// It stays hidden in local dev because blockers skip localhost, and appears
// the moment the dashboard is served from a real host. Do not rename it back.
//
// Events (name — meaning):
//   - Signup             — account created / signed in
//   - QuotaWallSeen      — the 402 wall modal opened
//   - UpsellModalSeen    — the voluntary upgrade modal opened
//   - QuotaWallCheckout  — a plan/top-up clicked inside the wall modal
//   - UpsellModalCheckout— a plan/top-up clicked inside the upsell modal
//   - CheckoutStarted    — checkout clicked, on ANY surface (see `source` prop:
//                          'wall' | 'upsell' | 'pricing')
//   - CheckoutRedirected — Stripe returned a URL and we are sending them there
//   - CheckoutFailed     — /api/billing/checkout errored, `reason` says why
//   - Subscribed         — plan activated after checkout
//   - SocialNudgeSeen    — post-generation "connect socials" banner rendered
//   - SocialNudgeConnect — its connect button clicked (opens hosted connect page)
//   - SocialNudgeDismissed — its X clicked (persisted, never shown again)
//   - ClipTutorialStarted  — first-login tutorial: user hit Start
//   - ClipTutorialSkipped  — first-login tutorial dismissed (intro or coach)
//   - ClipTutorialCompleted— first Clip Generator job finished with clips
// The Started → Redirected → Subscribed chain is what separates "never reached
// Stripe" from "reached Stripe and abandoned"; before 2-ago-2026 the modals
// emitted only their own *Checkout event and the difference was invisible.
// Prices ride along as ordinary props (e.g. value_usd) for breakdowns.
export function track(event, options) {
  try {
    if (typeof window !== 'undefined' && typeof window.op === 'function') {
      window.op('track', event, (options && options.props) || {});
    }
  } catch (_) {
    /* analytics must never throw into the app */
  }
}

/**
 * Bind this browser to the signed-in account. `profileId` is the user's uuid —
 * the same id the backend uses for its server-side events (ClipsDelivered,
 * JobFailed and the `revenue` mirror of the Stripe webhook) — so a sale lands
 * on the profile that carries the first visit's referrer and campaign.
 * OpenPanel keeps the profile in memory only, so this runs on every boot.
 */
export function identify(user, props) {
  try {
    if (!user || !user.id) return;
    if (typeof window !== 'undefined' && typeof window.op === 'function') {
      window.op('identify', { profileId: String(user.id), email: user.email, ...(props || {}) });
    }
  } catch (_) {
    /* analytics must never throw into the app */
  }
}

/** Forget the current profile (sign-out). */
export function reset() {
  try {
    if (typeof window !== 'undefined' && typeof window.op === 'function') {
      window.op('clear');
    }
  } catch (_) {
    /* ignore */
  }
}
