import { redirect } from "next/navigation";

export default async function ViewerPage({
  searchParams,
}: {
  searchParams: Promise<{ lab?: string | string[] }>;
}) {
  const params = await searchParams;
  const lab = Array.isArray(params.lab) ? params.lab[0] : params.lab;
  redirect(lab ? `/?lab=${encodeURIComponent(lab)}` : "/");
}
