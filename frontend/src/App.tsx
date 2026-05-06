import { useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  AlertCircle,
  Calculator,
  CheckCircle2,
  Divide,
  GitBranch,
  Grid3X3,
  Loader2,
  Moon,
  Parentheses,
  Pi,
  Play,
  Plus,
  Radical,
  Repeat2,
  RotateCcw,
  Sun,
  Superscript,
  Table2,
  Trash2,
  Workflow,
  X,
  type LucideIcon,
} from "lucide-react";

import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Textarea } from "./components/ui/textarea";
import { cn } from "./lib/utils";

type MethodKey =
  | "bisection"
  | "secant"
  | "iteration"
  | "newton"
  | "jacobi"
  | "gauss_seidel"
  | "lagrange"
  | "nfdf"
  | "nbdf"
  | "nfddf"
  | "nbddf";

type ResultPoint = {
  x: number;
  y: number;
  source?: "given" | "recovered";
};

type TableRow = {
  index: number;
  x: number;
  values: number[];
};

type IterationBranch = {
  branch: number;
  gExpression: string;
  fixedPoint?: number;
  iterations: number;
  converged: boolean;
  error?: number | null;
  failure?: string;
};

type ApiResult = {
  method: string;
  root?: number;
  fixedPoint?: number;
  value?: number;
  solution?: number[];
  polynomial?: string;
  target?: number;
  iterations?: number;
  converged?: boolean;
  residual?: number;
  warning?: string | null;
  derivativeMode?: string;
  secondDerivativeMode?: string;
  normalizedExpression?: string;
  formula?: string;
  convergenceStatus?: string;
  convergenceMessage?: string;
  initialDerivative?: number;
  initialSecondDerivative?: number;
  initialConvergenceProduct?: number;
  selectedBranch?: number;
  selectedGExpression?: string;
  branches?: IterationBranch[];
  solverMode?: "iterative" | "reordered_iterative" | "direct_fallback";
  rowOrder?: number[];
  mode?: "evaluate" | "recover_missing";
  points?: ResultPoint[];
  knownPoints?: ResultPoint[];
  missing?: { index: number; x: number; value: number } | null;
  table?: TableRow[];
  tableKind?: "basis" | "finite_difference" | "divided_difference";
  h?: number;
  p?: number;
  steps: Record<string, unknown>[];
};

type Point = { x: string; y: string };
type ThemeMode = "light" | "dark";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:5050";
const interpolationMethodKeys = new Set<MethodKey>(["lagrange", "nfdf", "nbdf", "nfddf", "nbddf"]);

function isInterpolationMethodKey(value: MethodKey) {
  return interpolationMethodKeys.has(value);
}

function getInitialTheme(): ThemeMode {
  if (typeof window === "undefined") {
    return "light";
  }

  const storedTheme = window.localStorage.getItem("theme");
  if (storedTheme === "light" || storedTheme === "dark") {
    return storedTheme;
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

const methods: Array<{
  key: MethodKey;
  title: string;
  short: string;
  accent: string;
  icon: typeof Calculator;
}> = [
  {
    key: "bisection",
    title: "Bisection Method",
    short: "Bracketed root",
    accent: "bg-sky-100 text-sky-700 dark:bg-sky-900 dark:text-sky-100",
    icon: GitBranch,
  },
  {
    key: "secant",
    title: "Secant Method",
    short: "Two-point root",
    accent: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-100",
    icon: Workflow,
  },
  {
    key: "iteration",
    title: "Method of Simple Iteration",
    short: "Fixed point",
    accent: "bg-violet-100 text-violet-700 dark:bg-violet-900 dark:text-violet-100",
    icon: Repeat2,
  },
  {
    key: "newton",
    title: "Newton-Raphson Method",
    short: "Derivative root",
    accent: "bg-cyan-100 text-cyan-700 dark:bg-cyan-900 dark:text-cyan-100",
    icon: Calculator,
  },
  {
    key: "jacobi",
    title: "Jacobi Method",
    short: "Linear system",
    accent: "bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-100",
    icon: Grid3X3,
  },
  {
    key: "gauss_seidel",
    title: "Gauss-Seidel Method",
    short: "Linear system",
    accent: "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-100",
    icon: Table2,
  },
  {
    key: "lagrange",
    title: "Lagrange Interpolation",
    short: "Polynomial fit",
    accent: "bg-teal-100 text-teal-700 dark:bg-teal-900 dark:text-teal-100",
    icon: Plus,
  },
  {
    key: "nfdf",
    title: "NFDF",
    short: "Newton forward difference",
    accent: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-100",
    icon: Workflow,
  },
  {
    key: "nbdf",
    title: "NBDF",
    short: "Newton backward difference",
    accent: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-100",
    icon: Workflow,
  },
  {
    key: "nfddf",
    title: "NFDDF",
    short: "Forward divided difference",
    accent: "bg-rose-100 text-rose-700 dark:bg-rose-900 dark:text-rose-100",
    icon: Table2,
  },
  {
    key: "nbddf",
    title: "NBDDF",
    short: "Backward divided difference",
    accent: "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-100",
    icon: Table2,
  },
];

const initialMatrix = [
  ["10", "-1", "2"],
  ["-1", "11", "-1"],
  ["2", "-1", "10"],
];

const initialVector = ["6", "25", "-11"];
const initialGuess = ["0", "0", "0"];
const mathFunctions = ["sin", "cos", "tan", "sqrt", "ln", "log", "exp", "abs"] as const;
const previewFunctions = [...mathFunctions, "asin", "acos", "atan", "sinh", "cosh", "tanh"].sort((a, b) => b.length - a.length);

function parseNumber(value: string, label: string) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`${label} must be a valid number.`);
  }
  return number;
}

function formatNumber(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return String(value ?? "");
  }
  const rounded = Number(value.toPrecision(9));
  const magnitude = Math.abs(rounded);
  if (magnitude >= 100000) {
    return rounded.toExponential(6);
  }
  if (magnitude > 0 && magnitude < 0.0001) {
    const decimalPlaces = Math.min(18, Math.max(6, Math.ceil(-Math.log10(magnitude)) + 4));
    return rounded.toFixed(decimalPlaces).replace(/0+$/, "").replace(/\.$/, "");
  }
  return rounded.toString();
}

function formatVector(values: unknown) {
  if (!Array.isArray(values)) {
    return String(values ?? "");
  }
  return `[${values.map((value) => formatNumber(value)).join(", ")}]`;
}

function focusNextInput(event: KeyboardEvent<HTMLInputElement>) {
  if (event.key !== "Enter" || event.nativeEvent.isComposing) {
    return;
  }

  const form = event.currentTarget.form;
  if (!form) {
    return;
  }

  const inputs = Array.from(form.querySelectorAll<HTMLInputElement>("input:not([disabled]):not([type='hidden'])"));
  const currentIndex = inputs.indexOf(event.currentTarget);
  const nextInput = inputs[currentIndex + 1];

  if (!nextInput) {
    return;
  }

  event.preventDefault();
  nextInput.focus();
  nextInput.select();
}

function Field({
  label,
  children,
  className,
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("space-y-2", className)}>
      <Label>{label}</Label>
      {children}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-h-[86px] rounded-md border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-950">
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-300">{label}</div>
      <div className="mt-2 break-words font-mono text-lg font-semibold text-slate-950 dark:text-slate-50">{value}</div>
    </div>
  );
}

function isNameBoundary(value: string, index: number) {
  return index < 0 || index >= value.length || !/[A-Za-z0-9_]/.test(value[index]);
}

function formatEquationText(value: string) {
  return value
    .replace(/\bpi\b/g, "π")
    .replace(/\*/g, "·")
    .replace(/\//g, " / ")
    .replace(/-/g, "−");
}

function readExponent(value: string, startIndex: number) {
  if (value[startIndex] === "(") {
    let depth = 0;
    for (let index = startIndex; index < value.length; index += 1) {
      if (value[index] === "(") {
        depth += 1;
      }
      if (value[index] === ")") {
        depth -= 1;
        if (depth === 0) {
          return { text: value.slice(startIndex + 1, index), endIndex: index + 1 };
        }
      }
    }
  }

  let endIndex = startIndex;
  while (endIndex < value.length && /[A-Za-z0-9_.]/.test(value[endIndex])) {
    endIndex += 1;
  }

  if (endIndex === startIndex && startIndex < value.length) {
    endIndex += 1;
  }

  return { text: value.slice(startIndex, endIndex), endIndex };
}

function renderEquationPreview(value: string) {
  const source = value.trim();

  if (!source) {
    return <span className="text-slate-400 dark:text-slate-500">...</span>;
  }

  const nodes: ReactNode[] = [];

  for (let index = 0; index < source.length; ) {
    const char = source[index];
    const exponentStart = char === "^" ? index + 1 : char === "*" && source[index + 1] === "*" ? index + 2 : -1;

    if (exponentStart > -1) {
      const exponent = readExponent(source, exponentStart);
      nodes.push(
        <sup key={`exp-${index}`} className="ml-0.5 text-[0.72em] font-semibold leading-none text-blue-600 dark:text-sky-300">
          {formatEquationText(exponent.text)}
        </sup>,
      );
      index = exponent.endIndex;
      continue;
    }

    const matchedFunction = previewFunctions.find(
      (name) => source.startsWith(name, index) && isNameBoundary(source, index - 1) && isNameBoundary(source, index + name.length),
    );

    if (matchedFunction) {
      nodes.push(
        <span key={`fn-${index}`} className="font-semibold text-sky-600 dark:text-cyan-300">
          {matchedFunction}
        </span>,
      );
      index += matchedFunction.length;
      continue;
    }

    if (source.startsWith("pi", index) && isNameBoundary(source, index - 1) && isNameBoundary(source, index + 2)) {
      nodes.push(
        <span key={`pi-${index}`} className="font-semibold text-violet-600 dark:text-violet-300">
          π
        </span>,
      );
      index += 2;
      continue;
    }

    if (char === "*") {
      nodes.push(
        <span key={`mul-${index}`} className="mx-1 text-slate-500 dark:text-slate-300">
          ·
        </span>,
      );
      index += 1;
      continue;
    }

    if (char === "/") {
      nodes.push(
        <span key={`div-${index}`} className="mx-1.5 inline-flex h-7 items-center text-lg font-semibold text-slate-500 dark:text-slate-300">
          /
        </span>,
      );
      index += 1;
      continue;
    }

    if (char === "-") {
      nodes.push(
        <span key={`minus-${index}`} className="mx-0.5 text-slate-500 dark:text-slate-300">
          −
        </span>,
      );
      index += 1;
      continue;
    }

    if (char === "x") {
      nodes.push(
        <span key={`x-${index}`} className="font-serif italic text-sky-700 dark:text-sky-200">
          x
        </span>,
      );
      index += 1;
      continue;
    }

    nodes.push(<span key={`char-${index}`}>{char}</span>);
    index += 1;
  }

  return nodes;
}

function EquationEditor({
  label,
  prefix,
  value,
  onChange,
  compact = false,
}: {
  label: string;
  prefix: string;
  value: string;
  onChange: (nextValue: string) => void;
  compact?: boolean;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function insertSnippet(snippet: string, cursorOffset = snippet.length) {
    const textarea = textareaRef.current;
    const start = textarea?.selectionStart ?? value.length;
    const end = textarea?.selectionEnd ?? value.length;
    const selectedText = value.slice(start, end);
    const nextSnippet = snippet.includes("$") ? snippet.replace("$", selectedText) : snippet;
    const nextValue = `${value.slice(0, start)}${nextSnippet}${value.slice(end)}`;
    const selectionStart = start + Math.max(0, cursorOffset);

    onChange(nextValue);
    window.requestAnimationFrame(() => {
      textarea?.focus();
      textarea?.setSelectionRange(selectionStart, selectionStart);
    });
  }

  function insertPower() {
    const textarea = textareaRef.current;
    const start = textarea?.selectionStart ?? value.length;
    const end = textarea?.selectionEnd ?? value.length;
    const selectedText = value.slice(start, end);
    const nextSnippet = selectedText ? `${selectedText}^()` : "^()";
    const nextValue = `${value.slice(0, start)}${nextSnippet}${value.slice(end)}`;
    const selectionStart = start + nextSnippet.length - 1;

    onChange(nextValue);
    window.requestAnimationFrame(() => {
      textarea?.focus();
      textarea?.setSelectionRange(selectionStart, selectionStart);
    });
  }

  function wrapSelection(before: string, after: string) {
    const textarea = textareaRef.current;
    const start = textarea?.selectionStart ?? value.length;
    const end = textarea?.selectionEnd ?? value.length;
    const selectedText = value.slice(start, end);
    const nextValue = `${value.slice(0, start)}${before}${selectedText}${after}${value.slice(end)}`;
    const selectionStart = selectedText ? start + before.length + selectedText.length + after.length : start + before.length;

    onChange(nextValue);
    window.requestAnimationFrame(() => {
      textarea?.focus();
      textarea?.setSelectionRange(selectionStart, selectionStart);
    });
  }

  function insertFunction(name: string) {
    wrapSelection(`${name}(`, ")");
  }

  const operatorButtons: Array<{
    label: string;
    title: string;
    icon?: LucideIcon;
    action: () => void;
  }> = [
    { label: "x", title: "Variable x", icon: X, action: () => insertSnippet("x") },
    { label: "x^n", title: "Power", icon: Superscript, action: insertPower },
    { label: "/", title: "Divide", icon: Divide, action: () => insertSnippet(" / ") },
    { label: "( )", title: "Parentheses", icon: Parentheses, action: () => wrapSelection("(", ")") },
    { label: "*", title: "Multiply", action: () => insertSnippet(" * ") },
    { label: "+", title: "Add", action: () => insertSnippet(" + ") },
    { label: "-", title: "Subtract", action: () => insertSnippet(" - ") },
  ];
  const constantButtons: Array<{ label: string; title: string; icon?: LucideIcon; action: () => void }> = [
    { label: "pi", title: "Pi", icon: Pi, action: () => insertSnippet("pi") },
    { label: "e", title: "Euler number", action: () => insertSnippet("e") },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <Label>{label}</Label>
        <Button type="button" variant="ghost" size="sm" onClick={() => onChange("")}>
          Clear
        </Button>
      </div>

      <div className="rounded-md border border-slate-200 bg-white p-4 shadow-sm shadow-slate-200/70 dark:border-slate-700 dark:bg-slate-900 dark:shadow-black/20">
        <div className="flex min-h-[58px] items-center gap-3 overflow-x-auto rounded-md border border-slate-300 bg-slate-50 px-4 py-3 dark:border-slate-600 dark:bg-slate-950">
          <span className="shrink-0 font-serif text-xl font-semibold italic text-slate-600 dark:text-slate-300">{prefix}</span>
          <span className="shrink-0 text-xl text-slate-600 dark:text-slate-300">=</span>
          <span className="min-w-0 whitespace-nowrap font-serif text-2xl leading-10 text-slate-950 dark:text-slate-50">
            {renderEquationPreview(value)}
          </span>
        </div>
      </div>

      <Textarea
        ref={textareaRef}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        spellCheck={false}
        className={cn("font-mono", compact ? "min-h-[76px]" : "min-h-[92px]")}
      />

      <div className="grid gap-2 rounded-md border border-slate-200 bg-slate-100 p-2 dark:border-slate-700 dark:bg-slate-950">
        <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
          {operatorButtons.map((button) => {
            const Icon = button.icon;
            return (
              <Button
                key={button.title}
                type="button"
                variant="outline"
                size="sm"
                onClick={button.action}
                title={button.title}
                aria-label={button.title}
                className="h-10 px-2 font-mono text-sm"
              >
                {Icon ? <Icon className="h-4 w-4" /> : null}
                <span>{button.label}</span>
              </Button>
            );
          })}
        </div>

        <div className="grid grid-cols-4 gap-2 sm:grid-cols-10">
          {mathFunctions.map((name) => (
            <Button
              key={name}
              type="button"
              variant="secondary"
              size="sm"
              onClick={() => insertFunction(name)}
              title={name}
              aria-label={name}
              className="h-10 px-2 font-mono text-sm"
            >
              {name === "sqrt" ? <Radical className="h-4 w-4" /> : null}
              {name}
            </Button>
          ))}
          {constantButtons.map((button) => {
            const Icon = button.icon;
            return (
              <Button
                key={button.title}
                type="button"
                variant="outline"
                size="sm"
                onClick={button.action}
                title={button.title}
                aria-label={button.title}
                className="h-10 px-2 font-mono text-sm"
              >
                {Icon ? <Icon className="h-4 w-4" /> : null}
                {button.label}
              </Button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);
  const [method, setMethod] = useState<MethodKey>("bisection");
  const [expression, setExpression] = useState("x^3 - x - 2");
  const [derivativeExpression, setDerivativeExpression] = useState("");
  const [gExpression, setGExpression] = useState("cos(x)");
  const [a, setA] = useState("1");
  const [b, setB] = useState("2");
  const [x0, setX0] = useState("1");
  const [x1, setX1] = useState("2");
  const [tolerance, setTolerance] = useState("0.000001");
  const [maxIterations, setMaxIterations] = useState("50");
  const [matrixSize, setMatrixSize] = useState(3);
  const [matrix, setMatrix] = useState(initialMatrix);
  const [vector, setVector] = useState(initialVector);
  const [initial, setInitial] = useState(initialGuess);
  const [points, setPoints] = useState<Point[]>([
    { x: "0", y: "1" },
    { x: "1", y: "3" },
    { x: "2", y: "2" },
  ]);
  const [target, setTarget] = useState("1.5");
  const [result, setResult] = useState<ApiResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const isDarkMode = theme === "dark";
  const selectedMethod = useMemo(() => methods.find((item) => item.key === method) ?? methods[0], [method]);
  const SelectedIcon = selectedMethod.icon;
  const isLinearMethod = method === "jacobi" || method === "gauss_seidel";
  const isInterpolationMethod = isInterpolationMethodKey(method);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDarkMode);
    window.localStorage.setItem("theme", theme);
  }, [isDarkMode, theme]);

  function resizeMatrix(nextSize: number) {
    const size = Math.min(6, Math.max(2, nextSize));
    setMatrixSize(size);
    setMatrix((current) =>
      Array.from({ length: size }, (_, rowIndex) =>
        Array.from({ length: size }, (_, columnIndex) => current[rowIndex]?.[columnIndex] ?? (rowIndex === columnIndex ? "1" : "0")),
      ),
    );
    setVector((current) => Array.from({ length: size }, (_, index) => current[index] ?? "0"));
    setInitial((current) => Array.from({ length: size }, (_, index) => current[index] ?? "0"));
  }

  function updateMatrix(rowIndex: number, columnIndex: number, value: string) {
    setMatrix((current) =>
      current.map((row, r) => row.map((cell, c) => (r === rowIndex && c === columnIndex ? value : cell))),
    );
  }

  function updatePoint(index: number, key: keyof Point, value: string) {
    setPoints((current) => current.map((point, pointIndex) => (pointIndex === index ? { ...point, [key]: value } : point)));
  }

  function resetExamples() {
    setExpression("x^3 - x - 2");
    setDerivativeExpression("");
    setGExpression("cos(x)");
    setA("1");
    setB("2");
    setX0("1");
    setX1("2");
    setTolerance("0.000001");
    setMaxIterations("50");
    setMatrixSize(3);
    setMatrix(initialMatrix);
    setVector(initialVector);
    setInitial(initialGuess);
    setPoints([
      { x: "0", y: "1" },
      { x: "1", y: "3" },
      { x: "2", y: "2" },
    ]);
    setTarget("1.5");
    setError("");
    setResult(null);
  }

  function buildPayload() {
    const common = {
      tolerance: parseNumber(tolerance, "Tolerance"),
      maxIterations: parseNumber(maxIterations, "Max iterations"),
    };

    if (method === "bisection") {
      return {
        method,
        params: {
          ...common,
          expression,
          a: parseNumber(a, "a"),
          b: parseNumber(b, "b"),
        },
      };
    }

    if (method === "secant") {
      return {
        method,
        params: {
          ...common,
          expression,
          x0: parseNumber(x0, "x0"),
          x1: parseNumber(x1, "x1"),
        },
      };
    }

    if (method === "iteration") {
      return {
        method,
        params: {
          ...common,
          gExpression,
          x0: parseNumber(x0, "x0"),
        },
      };
    }

    if (method === "newton") {
      return {
        method,
        params: {
          ...common,
          expression,
          derivativeExpression: derivativeExpression.trim() || undefined,
          x0: parseNumber(x0, "x0"),
        },
      };
    }

    if (isLinearMethod) {
      return {
        method,
        params: {
          ...common,
          matrix: matrix.map((row, rowIndex) => row.map((cell, columnIndex) => parseNumber(cell, `A${rowIndex + 1}${columnIndex + 1}`))),
          vector: vector.map((cell, index) => parseNumber(cell, `b${index + 1}`)),
          initial: initial.map((cell, index) => parseNumber(cell, `x0${index + 1}`)),
        },
      };
    }

    const interpolationPoints = points.map((point, index) => ({
      x: parseNumber(point.x, `x${index + 1}`),
      y: point.y.trim() === "" ? null : parseNumber(point.y, `f(x)${index + 1}`),
    }));
    const missingCount = interpolationPoints.filter((point) => point.y === null).length;
    if (missingCount > 1) {
      throw new Error("Only one f(x) value can be blank.");
    }

    const params: { points: Array<{ x: number; y: number | null }>; target?: number } = {
      points: interpolationPoints,
    };
    if (missingCount === 0) {
      params.target = parseNumber(target, "Target x");
    }

    return {
      method,
      params,
    };
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = buildPayload();
      const response = await fetch(`${API_URL}/api/calculate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await response.json();
      if (!response.ok || !json.ok) {
        throw new Error(json.error ?? "Calculation failed.");
      }
      setResult(json.data);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Calculation failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={cn("min-h-screen bg-slate-100 text-slate-950", isDarkMode && "dark bg-slate-950 text-slate-50")}>
      <header className="border-b border-slate-200 bg-white text-slate-950 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-50">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-md bg-sky-100 text-sky-700 dark:bg-sky-900 dark:text-sky-100">
                <Calculator className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-xl font-semibold tracking-normal sm:text-2xl">Numerical Methods Calculator</h1>
              </div>
            </div>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
            aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
            title={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
            className="w-full sm:w-auto"
          >
            {isDarkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            {isDarkMode ? "Light mode" : "Dark mode"}
          </Button>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[310px_minmax(0,1fr)] lg:px-8">
        <aside className="space-y-3">
          {methods.map((item) => {
            const Icon = item.icon;
            const active = item.key === method;
            return (
              <button
                key={item.key}
                type="button"
                onClick={() => {
                  setMethod(item.key);
                  setResult(null);
                  setError("");
                }}
                className={cn(
                  "flex min-h-[76px] w-full items-center gap-3 rounded-md border bg-white p-3 text-left shadow-sm shadow-slate-200/70 transition hover:-translate-y-0.5 hover:border-blue-500 hover:bg-slate-50 dark:bg-slate-900 dark:shadow-black/25 dark:hover:border-sky-300 dark:hover:bg-slate-800",
                  active ? "border-blue-500 ring-2 ring-blue-200 dark:border-sky-300 dark:ring-sky-500/35" : "border-slate-200 dark:border-slate-700",
                )}
              >
                <span className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-md", item.accent)}>
                  <Icon className="h-5 w-5" />
                </span>
                <span className="min-w-0">
                  <span className="block text-sm font-semibold text-slate-950 dark:text-slate-50">{item.title}</span>
                  <span className="mt-1 block text-xs text-slate-500 dark:text-slate-300">{item.short}</span>
                </span>
              </button>
            );
          })}
        </aside>

        <div className="space-y-6">
          <section className="rounded-md border border-slate-200 bg-white shadow-panel dark:border-slate-700 dark:bg-slate-900">
            <div className="flex flex-col gap-3 border-b border-slate-200 p-4 sm:flex-row sm:items-center sm:justify-between dark:border-slate-700">
              <div className="flex items-center gap-3">
                <span className={cn("flex h-11 w-11 items-center justify-center rounded-md", selectedMethod.accent)}>
                  <SelectedIcon className="h-5 w-5" />
                </span>
                <div>
                  <h2 className="text-lg font-semibold tracking-normal text-slate-950 dark:text-slate-50">{selectedMethod.title}</h2>
                  <p className="text-sm text-slate-500 dark:text-slate-300">{selectedMethod.short}</p>
                </div>
              </div>
              <Button type="button" variant="outline" size="sm" onClick={resetExamples}>
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5 p-4">
              <AnimatePresence mode="wait">
                <motion.div
                  key={method}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.18 }}
                  className="space-y-5"
                >
                  {renderMethodForm()}
                </motion.div>
              </AnimatePresence>

              {!isInterpolationMethod && (
                <div className="grid gap-4 sm:grid-cols-2">
                  <Field label="Tolerance">
                    <Input value={tolerance} onChange={(event) => setTolerance(event.target.value)} inputMode="decimal" />
                  </Field>
                  <Field label="Max iterations">
                    <Input value={maxIterations} onChange={(event) => setMaxIterations(event.target.value)} inputMode="numeric" />
                  </Field>
                </div>
              )}

              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <Button type="submit" disabled={loading} className="w-full sm:w-auto">
                  {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  Calculate
                </Button>
                <div className="min-h-[24px] text-sm text-slate-500 dark:text-slate-300">{result ? `${result.steps.length} step rows generated` : ""}</div>
              </div>
            </form>
          </section>

          <ResultPanel result={result} error={error} method={method} />
        </div>
      </main>
    </div>
  );

  function renderMethodForm() {
    if (method === "bisection") {
      return (
        <>
          <EquationEditor label="f(x)" prefix="f(x)" value={expression} onChange={setExpression} />
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="a">
              <Input value={a} onChange={(event) => setA(event.target.value)} inputMode="decimal" />
            </Field>
            <Field label="b">
              <Input value={b} onChange={(event) => setB(event.target.value)} inputMode="decimal" />
            </Field>
          </div>
        </>
      );
    }

    if (method === "secant") {
      return (
        <>
          <EquationEditor label="f(x)" prefix="f(x)" value={expression} onChange={setExpression} />
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="x0">
              <Input value={x0} onChange={(event) => setX0(event.target.value)} inputMode="decimal" />
            </Field>
            <Field label="x1">
              <Input value={x1} onChange={(event) => setX1(event.target.value)} inputMode="decimal" />
            </Field>
          </div>
        </>
      );
    }

    if (method === "iteration") {
      return (
        <>
          <EquationEditor label="Equation or g(x) candidates" prefix="g(x)" value={gExpression} onChange={setGExpression} />
          <Field label="x0">
            <Input value={x0} onChange={(event) => setX0(event.target.value)} inputMode="decimal" />
          </Field>
        </>
      );
    }

    if (method === "newton") {
      return (
        <>
          <EquationEditor label="Equation or f(x)" prefix="f(x)" value={expression} onChange={setExpression} />
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(220px,0.35fr)]">
            <Field label="f'(x) optional">
              <Input value={derivativeExpression} onChange={(event) => setDerivativeExpression(event.target.value)} spellCheck={false} className="font-mono" />
            </Field>
            <Field label="x0">
              <Input value={x0} onChange={(event) => setX0(event.target.value)} inputMode="decimal" />
            </Field>
          </div>
        </>
      );
    }

    if (isLinearMethod) {
      return renderLinearSystemEditor();
    }

    return renderInterpolationEditor();
  }

  function renderLinearSystemEditor() {
    return (
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Label>System size</Label>
          <div className="flex items-center gap-2">
            <Button type="button" variant="outline" size="sm" onClick={() => resizeMatrix(matrixSize - 1)}>
              -1
            </Button>
            <div className="flex h-8 min-w-[72px] items-center justify-center rounded-md border border-slate-300 bg-slate-50 text-sm font-semibold text-slate-800 dark:border-slate-600 dark:bg-slate-950 dark:text-slate-100">
              {matrixSize} x {matrixSize}
            </div>
            <Button type="button" variant="outline" size="sm" onClick={() => resizeMatrix(matrixSize + 1)}>
              +1
            </Button>
          </div>
        </div>

        <div className="overflow-x-auto rounded-md border border-slate-200 bg-slate-100 dark:border-slate-700 dark:bg-slate-950">
          <div className="min-w-[620px] p-3">
            <div
              className="grid gap-2"
              style={{ gridTemplateColumns: `repeat(${matrixSize}, minmax(72px, 1fr)) 24px minmax(72px, 1fr)` }}
            >
              {matrix.map((row, rowIndex) => (
                <div key={`matrix-row-${rowIndex}`} className="contents">
                  {row.map((cell, columnIndex) => (
                    <Input
                      key={`${rowIndex}-${columnIndex}`}
                      aria-label={`A ${rowIndex + 1} ${columnIndex + 1}`}
                      value={cell}
                      onChange={(event) => updateMatrix(rowIndex, columnIndex, event.target.value)}
                      onKeyDown={focusNextInput}
                      className="h-9 text-center font-mono"
                    />
                  ))}
                  <div className="flex items-center justify-center text-sm font-semibold text-slate-500 dark:text-slate-300">=</div>
                  <Input
                    aria-label={`b ${rowIndex + 1}`}
                    value={vector[rowIndex]}
                    onChange={(event) => setVector((current) => current.map((item, itemIndex) => (itemIndex === rowIndex ? event.target.value : item)))}
                    onKeyDown={focusNextInput}
                    className="h-9 text-center font-mono"
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-6">
          {initial.map((cell, index) => (
            <Field key={index} label={`x0 ${index + 1}`}>
              <Input
                value={cell}
                onChange={(event) => setInitial((current) => current.map((item, itemIndex) => (itemIndex === index ? event.target.value : item)))}
                onKeyDown={focusNextInput}
                className="font-mono"
              />
            </Field>
          ))}
        </div>
      </div>
    );
  }

  function renderInterpolationEditor() {
    const missingPoint = points.find((point) => point.y.trim() === "");

    return (
      <div className="space-y-4">
        <div className="grid gap-3">
          <div className="grid grid-cols-[1fr_1fr_40px] gap-2 px-1 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-300">
            <span>x</span>
            <span>f(x)</span>
            <span />
          </div>
          {points.map((point, index) => (
            <div key={index} className="grid grid-cols-[1fr_1fr_40px] gap-2">
              <Input
                aria-label={`x ${index + 1}`}
                value={point.x}
                onChange={(event) => updatePoint(index, "x", event.target.value)}
                onKeyDown={focusNextInput}
                inputMode="decimal"
              />
              <Input
                aria-label={`f(x) ${index + 1}`}
                value={point.y}
                onChange={(event) => updatePoint(index, "y", event.target.value)}
                onKeyDown={focusNextInput}
                inputMode="decimal"
                placeholder="missing"
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={() => setPoints((current) => current.filter((_, pointIndex) => pointIndex !== index))}
                disabled={points.length <= 2}
                aria-label="Remove point"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>
        <div className="flex flex-col gap-3 sm:flex-row">
          <Button type="button" variant="secondary" onClick={() => setPoints((current) => [...current, { x: "", y: "" }])}>
            <Plus className="h-4 w-4" />
            Add point
          </Button>
          <Field label={missingPoint ? "Missing x" : "Target x"} className="sm:min-w-[220px]">
            <Input
              value={missingPoint ? missingPoint.x : target}
              onChange={(event) => setTarget(event.target.value)}
              onKeyDown={focusNextInput}
              inputMode="decimal"
              disabled={Boolean(missingPoint)}
            />
          </Field>
        </div>
      </div>
    );
  }
}

function ResultPanel({ result, error, method }: { result: ApiResult | null; error: string; method: MethodKey }) {
  if (error) {
    return (
      <section className="rounded-md border border-red-200 bg-red-50 p-4 text-red-800 dark:border-red-700 dark:bg-red-950 dark:text-red-100">
        <div className="flex items-start gap-3">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
          <div>
            <h2 className="font-semibold">Calculation error</h2>
            <p className="mt-1 text-sm">{error}</p>
          </div>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="rounded-md border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-500 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-300">
        No result
      </section>
    );
  }

  const isInterpolation = isInterpolationMethodKey(method);
  const primaryValue =
    result.solution !== undefined
      ? formatVector(result.solution)
      : result.fixedPoint !== undefined
        ? formatNumber(result.fixedPoint)
        : result.root !== undefined
          ? formatNumber(result.root)
          : formatNumber(result.value);

  return (
    <section className="space-y-4">
      <div className="rounded-md border border-slate-200 bg-white p-4 shadow-panel dark:border-slate-700 dark:bg-slate-900">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-950 dark:text-slate-50">{result.method}</h2>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-slate-500 dark:text-slate-300">
              {result.mode === "recover_missing" && (
                <span className="inline-flex items-center gap-1 rounded-md bg-emerald-100 px-2 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-900 dark:text-emerald-100">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Missing f(x) recovered
                </span>
              )}
              {method === "newton" && result.derivativeMode && (
                <span className="inline-flex items-center gap-1 rounded-md bg-sky-100 px-2 py-1 text-xs font-semibold text-sky-700 dark:bg-sky-900 dark:text-sky-100">
                  {result.derivativeMode === "manual" ? "Manual f'(x)" : "Auto f'(x)"}
                </span>
              )}
              {method === "newton" && result.secondDerivativeMode === "auto" && (
                <span className="inline-flex items-center gap-1 rounded-md bg-sky-100 px-2 py-1 text-xs font-semibold text-sky-700 dark:bg-sky-900 dark:text-sky-100">
                  Auto f''(x)
                </span>
              )}
              {method === "iteration" && result.selectedBranch !== undefined && result.branches && result.branches.length > 1 && (
                <span className="inline-flex items-center gap-1 rounded-md bg-sky-100 px-2 py-1 text-xs font-semibold text-sky-700 dark:bg-sky-900 dark:text-sky-100">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Branch {result.selectedBranch} selected
                </span>
              )}
              {result.solverMode === "direct_fallback" ? (
                <span className="inline-flex items-center gap-1 rounded-md bg-emerald-100 px-2 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-900 dark:text-emerald-100">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Direct solve
                </span>
              ) : typeof result.converged === "boolean" ? (
                <span className={cn("inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-semibold", result.converged ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-100" : "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-100")}>
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  {result.converged ? "Converged" : method === "newton" && result.convergenceStatus === "stalled" ? "Not converged" : "Max iterations"}
                </span>
              ) : null}
              {result.solverMode === "reordered_iterative" && (
                <span className="rounded-md bg-sky-100 px-2 py-1 text-xs font-semibold text-sky-700 dark:bg-sky-900 dark:text-sky-100">
                  Rows reordered
                </span>
              )}
              {result.warning && <span className="rounded-md bg-amber-100 px-2 py-1 text-xs font-semibold text-amber-800 dark:bg-amber-900 dark:text-amber-100">{result.warning}</span>}
            </div>
          </div>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <Metric label={result.solution ? "Solution" : result.missing ? "Recovered f(x)" : isInterpolation ? "P(x)" : method === "iteration" ? "Fixed point" : "Root"} value={primaryValue} />
          <Metric label={isInterpolation ? "Terms" : "Iterations"} value={result.iterations ?? result.steps.length} />
          <Metric label={isInterpolation ? "Target x" : result.residual !== undefined ? "Residual" : "Value"} value={formatNumber(isInterpolation ? result.target : result.residual ?? result.value ?? "")} />
        </div>

        {result.polynomial && (
          <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-950">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-300">Polynomial</div>
            <div className="mt-1 break-words font-mono text-sm text-slate-950 dark:text-slate-50">{result.polynomial}</div>
          </div>
        )}

        {method === "newton" && (result.formula || result.convergenceMessage) && (
          <div className="mt-3 grid gap-3 lg:grid-cols-2">
            {result.normalizedExpression && (
              <div className="rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-950">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-300">Solved as f(x)</div>
                <div className="mt-1 break-words font-mono text-sm text-slate-950 dark:text-slate-50">{result.normalizedExpression}</div>
              </div>
            )}
            {result.formula && (
              <div className="rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-950">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-300">Newton formula</div>
                <div className="mt-1 break-words font-mono text-sm text-slate-950 dark:text-slate-50">{result.formula}</div>
              </div>
            )}
            {result.convergenceMessage && (
              <div className="rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-950">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-300">Convergence validation</div>
                <div className="mt-1 text-sm text-slate-700 dark:text-slate-100">{result.convergenceMessage}</div>
              </div>
            )}
          </div>
        )}

        {method === "newton" && (result.initialDerivative !== undefined || result.initialSecondDerivative !== undefined) && (
          <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-950">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-300">Derivative check at x0</div>
            <div className="mt-2 grid gap-2 text-sm text-slate-700 sm:grid-cols-3 dark:text-slate-100">
              <span className="font-mono">f'(x0) = {formatNumber(result.initialDerivative)}</span>
              <span className="font-mono">f''(x0) = {formatNumber(result.initialSecondDerivative)}</span>
              <span className="font-mono">f(x0)f''(x0) = {formatNumber(result.initialConvergenceProduct)}</span>
            </div>
          </div>
        )}

        {method === "iteration" && result.selectedGExpression && (
          <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-950">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-300">Selected g(x)</div>
            <div className="mt-1 break-words font-mono text-sm text-slate-950 dark:text-slate-50">{result.selectedGExpression}</div>
          </div>
        )}

        {method === "iteration" && result.branches && result.branches.length > 1 && (
          <div className="mt-3 overflow-hidden rounded-md border border-slate-200 dark:border-slate-700">
            <div className="border-b border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-300">
              Tried branches
            </div>
            <div className="overflow-auto">
              <table className="w-full min-w-[760px] border-collapse text-left text-sm">
                <thead className="bg-slate-100 text-xs uppercase tracking-wide text-slate-600 dark:bg-slate-800 dark:text-slate-100">
                  <tr>
                    <th className="border-b border-slate-200 px-3 py-2 font-semibold dark:border-slate-700">Branch</th>
                    <th className="border-b border-slate-200 px-3 py-2 font-semibold dark:border-slate-700">g(x)</th>
                    <th className="border-b border-slate-200 px-3 py-2 font-semibold dark:border-slate-700">Status</th>
                    <th className="border-b border-slate-200 px-3 py-2 font-semibold dark:border-slate-700">Fixed point</th>
                    <th className="border-b border-slate-200 px-3 py-2 font-semibold dark:border-slate-700">Error</th>
                  </tr>
                </thead>
                <tbody>
                  {result.branches.map((branch) => {
                    const selected = branch.branch === result.selectedBranch;
                    const status = branch.failure ? branch.failure : branch.converged ? "Converged" : "Max iterations";
                    return (
                      <tr key={branch.branch} className={selected ? "bg-sky-50 dark:bg-sky-950" : "odd:bg-white even:bg-slate-50 dark:odd:bg-slate-900 dark:even:bg-slate-800/60"}>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-700 dark:border-slate-700 dark:text-slate-100">
                          {branch.branch}
                          {selected ? <span className="ml-2 rounded-md bg-sky-100 px-1.5 py-0.5 font-sans text-[11px] font-semibold text-sky-700 dark:bg-sky-900 dark:text-sky-100">selected</span> : null}
                        </td>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-700 dark:border-slate-700 dark:text-slate-100">{branch.gExpression}</td>
                        <td className="border-b border-slate-200 px-3 py-2 text-xs text-slate-700 dark:border-slate-700 dark:text-slate-100">{status}</td>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-700 dark:border-slate-700 dark:text-slate-100">{formatNumber(branch.fixedPoint ?? "")}</td>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-700 dark:border-slate-700 dark:text-slate-100">{formatNumber(branch.error ?? "")}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {isInterpolation && <InterpolationVisuals result={result} />}

      <div className="rounded-md border border-slate-200 bg-white shadow-panel dark:border-slate-700 dark:bg-slate-900">
        <div className="border-b border-slate-200 p-4 dark:border-slate-700">
          <h2 className="text-lg font-semibold text-slate-950 dark:text-slate-50">Steps</h2>
        </div>
        <div className="max-h-[540px] overflow-auto">
          <StepTable method={method} steps={result.steps} />
        </div>
      </div>
    </section>
  );
}

function InterpolationVisuals({ result }: { result: ApiResult }) {
  const hasDifferenceTable = Boolean((result.table?.length ?? 0) > 0 && result.tableKind !== "basis");

  if (!hasDifferenceTable) {
    return null;
  }

  return (
    <div className="rounded-md border border-slate-200 bg-white shadow-panel dark:border-slate-700 dark:bg-slate-900">
      <div className="border-b border-slate-200 p-4 dark:border-slate-700">
        <h2 className="text-lg font-semibold text-slate-950 dark:text-slate-50">Visuals</h2>
      </div>

      <div className="p-4">
        <DifferenceTableView rows={result.table ?? []} kind={result.tableKind} />
      </div>
    </div>
  );
}

function DifferenceTableView({ rows, kind }: { rows: TableRow[]; kind?: ApiResult["tableKind"] }) {
  if (!rows.length || kind === "basis") {
    return null;
  }

  const maxValues = Math.max(...rows.map((row) => row.values.length));
  const headers = Array.from({ length: maxValues }, (_, index) => {
    if (index === 0) {
      return "f(x)";
    }
    return kind === "finite_difference" ? `Δ${index}` : `DD${index}`;
  });

  return (
    <div>
      <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-300">
        {kind === "finite_difference" ? "Difference table" : "Divided difference table"}
      </div>
      <div className="overflow-auto rounded-md border border-slate-200 dark:border-slate-700">
        <table className="w-full min-w-[420px] border-collapse text-left text-xs">
          <thead className="bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-100">
            <tr>
              <th className="border-b border-slate-200 px-3 py-2 font-semibold dark:border-slate-700">x</th>
              {headers.map((header) => (
                <th key={header} className="border-b border-slate-200 px-3 py-2 font-semibold dark:border-slate-700">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.index}-${row.x}`} className="odd:bg-white even:bg-slate-50 dark:odd:bg-slate-900 dark:even:bg-slate-800/60">
                <td className="border-b border-slate-200 px-3 py-2 font-mono text-slate-700 dark:border-slate-700 dark:text-slate-100">{formatNumber(row.x)}</td>
                {headers.map((_header, index) => (
                  <td key={index} className="border-b border-slate-200 px-3 py-2 font-mono text-slate-700 dark:border-slate-700 dark:text-slate-100">
                    {row.values[index] === undefined ? "" : formatNumber(row.values[index])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StepTable({ method, steps }: { method: MethodKey; steps: Record<string, unknown>[] }) {
  if (steps.length === 0) {
    return <div className="p-4 text-sm text-slate-500 dark:text-slate-300">No iterations were required.</div>;
  }

  const columnsByMethod: Record<MethodKey, Array<{ label: string; key: string; vector?: boolean }>> = {
    bisection: [
      { label: "i", key: "iteration" },
      { label: "a", key: "a" },
      { label: "b", key: "b" },
      { label: "c", key: "c" },
      { label: "f(c)", key: "fC" },
      { label: "error", key: "error" },
    ],
    secant: [
      { label: "i", key: "iteration" },
      { label: "x(i-1)", key: "xPrevious" },
      { label: "x(i)", key: "xCurrent" },
      { label: "x(i+1)", key: "xNext" },
      { label: "f(x)", key: "fNext" },
      { label: "error", key: "error" },
    ],
    iteration: [
      { label: "i", key: "iteration" },
      { label: "x(i)", key: "xCurrent" },
      { label: "g(x)", key: "gValue" },
      { label: "x(i+1)", key: "xNext" },
      { label: "error", key: "error" },
    ],
    newton: [
      { label: "i0", key: "iteration" },
      { label: "xi", key: "xCurrent" },
      { label: "x(i+1)", key: "xNext" },
      { label: "abs(x(i+1)-xi)", key: "error" },
    ],
    jacobi: [
      { label: "i", key: "iteration" },
      { label: "previous", key: "previous", vector: true },
      { label: "current", key: "current", vector: true },
      { label: "error", key: "error" },
      { label: "residual", key: "residual" },
    ],
    gauss_seidel: [
      { label: "i", key: "iteration" },
      { label: "previous", key: "previous", vector: true },
      { label: "current", key: "current", vector: true },
      { label: "error", key: "error" },
      { label: "residual", key: "residual" },
    ],
    lagrange: [
      { label: "i", key: "iteration" },
      { label: "x_i", key: "x" },
      { label: "y_i", key: "y" },
      { label: "L_i(x) equation", key: "basisEquation" },
      { label: "L_i(target) value", key: "basisValueEquation" },
      { label: "y_i L_i(target)", key: "termEquation" },
      { label: "partial P(x)", key: "partial" },
    ],
    nfdf: [
      { label: "k", key: "order" },
      { label: "Δ^k", key: "difference" },
      { label: "p term", key: "pTerm" },
      { label: "coefficient", key: "coefficient" },
      { label: "term", key: "term" },
      { label: "partial", key: "partial" },
    ],
    nbdf: [
      { label: "k", key: "order" },
      { label: "∇^k", key: "difference" },
      { label: "p term", key: "pTerm" },
      { label: "coefficient", key: "coefficient" },
      { label: "term", key: "term" },
      { label: "partial", key: "partial" },
    ],
    nfddf: [
      { label: "k", key: "order" },
      { label: "x anchor", key: "xAnchor" },
      { label: "divided diff", key: "dividedDifference" },
      { label: "product", key: "product" },
      { label: "term", key: "term" },
      { label: "partial", key: "partial" },
    ],
    nbddf: [
      { label: "k", key: "order" },
      { label: "x anchor", key: "xAnchor" },
      { label: "divided diff", key: "dividedDifference" },
      { label: "product", key: "product" },
      { label: "term", key: "term" },
      { label: "partial", key: "partial" },
    ],
  };

  const columns = columnsByMethod[method];

  return (
    <table className={cn("w-full border-collapse text-left text-sm", method === "lagrange" ? "min-w-[1180px]" : "min-w-[720px]")}>
      <thead className="sticky top-0 z-10 bg-slate-100 text-xs uppercase tracking-wide text-slate-600 dark:bg-slate-800 dark:text-slate-100">
        <tr>
          {columns.map((column) => (
            <th key={column.key} className="border-b border-slate-200 px-4 py-3 font-semibold dark:border-slate-600">
              {column.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {steps.map((step, index) => (
          <tr key={index} className="odd:bg-white even:bg-slate-50 dark:odd:bg-slate-900 dark:even:bg-slate-800/60">
            {columns.map((column) => (
              <td key={column.key} className="border-b border-slate-200 px-4 py-3 font-mono text-xs text-slate-700 dark:border-slate-700 dark:text-slate-100">
                {column.vector ? formatVector(step[column.key]) : formatNumber(step[column.key])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
