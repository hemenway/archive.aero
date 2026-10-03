// Shallow updates; subscribers observe one coherent state per microtask.
export function createStore(initial) {
  let state = initial, queued = false, previous;
  const listeners = new Set();
  return {
    get: () => state,
    set(update) {
      const patch = typeof update === 'function' ? update(state) : update;
      if (!patch) return;
      if (!queued) previous = state;
      state = { ...state, ...patch };
      if (!queued) {
        queued = true;
        queueMicrotask(() => {
          queued = false;
          for (const listener of listeners) listener(state, previous);
        });
      }
    },
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
  };
}
