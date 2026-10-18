export function NoActiveSession() {
  return (
    <div className="flex h-full items-center justify-center p-4">
      <div className="rounded-2xl border-2 border-dashed border-gray-300 bg-white px-8 py-12 text-center shadow-sm">
        <p className="text-lg font-semibold text-gray-900">Нет активной сессии</p>
        <p className="mt-1 text-sm text-gray-500">Откройте страницу колод и запустите изучение</p>
      </div>
    </div>
  );
}
