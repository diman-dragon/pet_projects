// Access to Capacitor (native) with safe fallbacks for browser dev.
const cap = window.Capacitor;
export const isNative = !!cap?.isNativePlatform?.();
export const plugin = (name) => cap?.Plugins?.[name] ?? (isNative ? cap.registerPlugin?.(name) : null) ?? null;
export const nativeUrl = (uri) => (cap?.convertFileSrc ? cap.convertFileSrc(uri) : uri);
