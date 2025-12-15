"use client";

export function Header() {
  return (
    <header className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 max-w-screen-2xl items-center">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground font-bold text-sm">
            A
          </div>
          <div className="flex flex-col">
            <h1 className="text-lg font-semibold leading-none">Autosteer</h1>
            <p className="text-xs text-muted-foreground leading-none mt-0.5">
              mechanistic interpretability
            </p>
          </div>
        </div>
      </div>
    </header>
  );
}
