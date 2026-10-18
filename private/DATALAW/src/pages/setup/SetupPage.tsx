import { DorlaneLogo } from '../../components/common/DorlaneLogo';
import { PublicLanguageSwitcher } from '../../components/auth/PublicLanguageSwitcher';
import { SetupWizard } from '../../components/setup/SetupWizard';

export default function SetupPage({ onComplete }: { onComplete: () => void }) {
  return (
    <div className="fixed inset-0 z-[1000] flex select-none items-center justify-center overflow-y-auto bg-[var(--app-modal-scrim)] p-6 font-[var(--font-ui)]">
      <div className="pointer-events-none absolute inset-0 backdrop-blur-xl" aria-hidden />
      <div className="relative z-10 flex w-full max-w-[520px] flex-col items-center gap-5">
        <DorlaneLogo size="lg" showText className="gap-4" />
        <PublicLanguageSwitcher />
        <SetupWizard onComplete={onComplete} />
      </div>
    </div>
  );
}
