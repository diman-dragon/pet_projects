import { Card } from "../../ui/Card";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  accent?: boolean;
}

export function StatCard({ title, value, subtitle, accent }: StatCardProps) {
  return (
    <Card className={`transition-all hover:shadow-md ${accent ? "bg-gradient-to-br from-[#0078D4] to-[#005A9E]" : "border border-gray-200"}`}>
      <div className={`text-sm font-semibold ${accent ? "text-blue-100" : "text-gray-500"}`}>
        {title}
      </div>
      <div className={`mt-3 text-3xl font-bold ${accent ? "text-white" : "text-gray-900"}`}>
        {value}
      </div>
      <div className={`mt-1 text-sm ${accent ? "text-blue-100" : "text-gray-500"}`}>
        {subtitle}
      </div>
    </Card>
  );
}
