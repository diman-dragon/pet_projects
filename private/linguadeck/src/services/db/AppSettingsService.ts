import type { AppSettings } from "../../types/domain";
import { invokeDb } from "./invokeDb";

export const AppSettingsService = {
  get() {
    return invokeDb<AppSettings>("settings_get");
  },

  save(settings: AppSettings) {
    return invokeDb<AppSettings>("settings_save", { settings });
  }
};
