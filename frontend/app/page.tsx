import { Header } from "@/components/header";
import { ThemeToggle } from "@/components/theme-toggle";
import { GenerationInterface } from "@/components/generation-interface";
import { SteeringInterface } from "@/components/steering-interface";

export default function Page() {
  return (
    <>
      <Header />
      <div className="container mx-auto max-w-5xl space-y-8 px-4 py-8">
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">AutoSteer</h1>
          <p className="text-lg text-muted-foreground">
            Explore and steer language model behavior using Sparse Autoencoders
          </p>
        </div>

        <div className="space-y-12">
          <section>
            <h2 className="mb-4 text-2xl font-semibold">Explore Features</h2>
            <GenerationInterface />
          </section>

          <section>
            <h2 className="mb-4 text-2xl font-semibold">Steer Generation</h2>
            <SteeringInterface />
          </section>
        </div>
      </div>
      <ThemeToggle />
    </>
  );
}
