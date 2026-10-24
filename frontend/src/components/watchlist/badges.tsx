import { cn } from "@/lib/utils";

/** Board → short tag. Unknown boards fall back to the raw value. */
export function boardTag(board?: string | null): { label: string; tone: string } {
  switch (board) {
    case "main_sh":
    case "SH":
      return { label: "Main SH", tone: "bg-muted text-muted-foreground" };
    case "main_sz":
    case "SZ":
      return { label: "Main SZ", tone: "bg-muted text-muted-foreground" };
    case "star":
    case "科创板":
      return { label: "STAR", tone: "bg-primary/10 text-primary" };
    case "chinext":
    case "创业板":
      return { label: "ChiNext", tone: "bg-warning/10 text-warning" };
    default:
      return { label: board ?? "—", tone: "bg-muted text-muted-foreground" };
  }
}

export function subScore(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(1);
}

export function StatusPill({
  label,
  tone,
}: {
  label: string;
  tone: "success" | "danger" | "warning" | "neutral" | "primary";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-2 py-0.5 text-xs font-medium",
        tone === "success" && "bg-success/10 text-success",
        tone === "danger" && "bg-danger/10 text-danger",
        tone === "warning" && "bg-warning/10 text-warning",
        tone === "primary" && "bg-primary/10 text-primary",
        tone === "neutral" && "bg-muted text-muted-foreground",
      )}
    >
      {label}
    </span>
  );
}
