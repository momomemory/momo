interface JsonViewProps {
  label: string;
  value: unknown;
}

export function JsonView({ label, value }: JsonViewProps) {
  return (
    <section class="panel">
      <h3>{label}</h3>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}
