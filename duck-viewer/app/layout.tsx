import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Microduck Studio",
  description: "Train, evaluate, inspect, and export Microduck policies locally",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
