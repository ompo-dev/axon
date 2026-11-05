import { useEffect, useRef, type ReactNode, type CSSProperties } from 'react';
import { motion } from 'motion/react';
import { X, SlidersHorizontal, Maximize2, MoreVertical, type LucideIcon } from 'lucide-react';

export function IconButton({
  icon: Icon,
  label,
  onClick,
  disabled = false,
  className = '',
}: {
  icon: LucideIcon;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      className={`icon-button ${className}`}
      title={label}
      aria-label={label}
      onClick={onClick}
      disabled={disabled}
    >
      <Icon size={17} strokeWidth={1.7} />
    </button>
  );
}
export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: string }) {
  return (
    <span className={`badge ${tone} inline-flex items-center`}>
      <i />
      {children}
    </span>
  );
}
export function Segments<T extends string>({
  items,
  value,
  onChange,
  label,
}: {
  items: readonly T[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segments" role="group" aria-label={label}>
      {items.map((item) => (
        <button
          key={item}
          type="button"
          aria-pressed={value === item}
          className={value === item ? 'selected' : ''}
          onClick={() => onChange(item)}
        >
          {item}
        </button>
      ))}
    </div>
  );
}
export function Ticks({
  value,
  total = 30,
  color = 'var(--blue)',
  label,
}: {
  value: number;
  total?: number;
  color?: string;
  label: string;
}) {
  return (
    <div
      className="ticks"
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={total}
      aria-valuenow={Math.max(0, Math.min(total, value))}
    >
      {Array.from({ length: total }, (_, i) => (
        <i key={i} style={{ background: i < value ? color : undefined }} />
      ))}
    </div>
  );
}
export function Sparkline({ values, color = 'var(--blue)' }: { values: number[]; color?: string }) {
  const lo = Math.min(...values),
    hi = Math.max(...values),
    points = values
      .map(
        (v, i) =>
          `${4 + (i / Math.max(1, values.length - 1)) * 170},${values.length === 1 ? 25 : 42 - ((v - lo) / Math.max(1e-12, hi - lo)) * 34}`,
      )
      .join(' ');
  return (
    <svg
      viewBox="0 0 180 50"
      className="sparkline"
      role="img"
      aria-label={`${values.length} amostras medidas`}
    >
      <path d="M4 43H175" stroke="var(--line)" />
      {values.length > 0 && (
        <>
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <circle
            cx={4 + (values.length > 1 ? 170 : 0)}
            cy={
              values.length === 1
                ? 25
                : 42 - ((values.at(-1)! - lo) / Math.max(1e-12, hi - lo)) * 34
            }
            r="2.5"
            fill={color}
          />
        </>
      )}
    </svg>
  );
}
export function Widget({
  title,
  sub,
  tools,
  children,
  footer,
  onInspect,
  onClose,
  className = '',
}: {
  title: string;
  sub?: string;
  tools?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  onInspect?: () => void;
  onClose?: () => void;
  className?: string;
}) {
  return (
    <motion.section
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`widget ${className}`}
    >
      <header className="widget-header">
        <h2>{title}</h2>
        {sub && <span className="widget-sub">{sub}</span>}
        <div className="widget-tools flex items-center">
          {tools}
          {onInspect && (
            <IconButton
              icon={SlidersHorizontal}
              label={`Inspecionar ${title}`}
              onClick={onInspect}
            />
          )}{' '}
          {onClose && <IconButton icon={X} label={`Ocultar ${title}`} onClick={onClose} />}
        </div>
      </header>
      <div className="widget-sections">{children}</div>
      {footer && (
        <footer className="widget-footer">
          {onInspect && (
            <>
              <IconButton icon={MoreVertical} label={`Dados de ${title}`} onClick={onInspect} />
              <span className="footer-divider" />
            </>
          )}
          {footer}
        </footer>
      )}
    </motion.section>
  );
}
export function Modal({
  title,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const d = ref.current!;
    d.showModal();
    d.querySelector<HTMLElement>('input:not([type=checkbox]), select, textarea')?.focus();
    const cancel = (e: Event) => {
      e.preventDefault();
      closeRef.current();
    };
    d.addEventListener('cancel', cancel);
    return () => {
      d.removeEventListener('cancel', cancel);
      d.close();
    };
  }, []);
  return (
    <dialog
      ref={ref}
      className={`modal ${wide ? 'wide' : ''}`}
      aria-label={title}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
    >
      <header>
        <h2>{title}</h2>
        <IconButton icon={X} label="Fechar" onClick={onClose} />
      </header>
      <div className="modal-content">{children}</div>
    </dialog>
  );
}
export function Empty({ icon: Icon, title }: { icon: LucideIcon; title: string }) {
  return (
    <div className="empty">
      <Icon size={23} strokeWidth={1.4} />
      <span>{title}</span>
    </div>
  );
}
export const colored = (color: string): CSSProperties => ({ '--accent': color }) as CSSProperties;
export function Expand({ onClick }: { onClick: () => void }) {
  return <IconButton icon={Maximize2} label="Ampliar inspecao" onClick={onClick} />;
}
