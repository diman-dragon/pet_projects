export const APPEAL_STATUS_OPTIONS = [
    { labelKey: 'status.draft', value: 'DRAFT' },
    { labelKey: 'status.ready_to_send', value: 'READY' },
    { labelKey: 'status.sent', value: 'SENT' },
    { labelKey: 'status.registered', value: 'REGISTERED' },
    { labelKey: 'status.received', value: 'RECEIVED' },
    { labelKey: 'status.closed', value: 'CLOSED' },
    { labelKey: 'status.archive', value: 'ARCHIVE' },
] as const;

export type AppealStatusOption = (typeof APPEAL_STATUS_OPTIONS)[number];
