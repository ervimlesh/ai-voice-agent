import type { PropsWithChildren } from 'react';

export function AppShell({ children }: PropsWithChildren) {
  return (
    <>
     <main className="app-shell">
     

      {children}
    </main>
    </>
   
  );
}
