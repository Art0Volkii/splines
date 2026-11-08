#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""АЛГОРИТМ 1. Кубічна B-сплайнова інтерполяція / відновлення зображень.

Режими:
  exact     — всі відліки, точна інтерполяція; метод прогонки,
              дзеркальні крайові умови, перевірка E_int.
  threshold — маска I[i,j] > tau, регуляризоване найменших квадратів.
  adaptive  — початкові 4 кутові точки, послідовне додавання точки
              найбільшої похибки, та сама регуляризована модель.

ОСНОВА: [1], [4] для exact; ідея порога — [2].
Регуляризовані режими — запропоновані в роботі чисельні модифікації,
НЕ дослівні алгоритми з [2] чи [3].

Встановлення: python -m pip install numpy scipy pillow matplotlib
Приклади:
  python algorithm_1_bspline.py --self-test
  python algorithm_1_bspline.py --input photo.jpg --mode exact --scale 2
  python algorithm_1_bspline.py --input photo.jpg --mode threshold --tau 0.4
  python algorithm_1_bspline.py --input photo.jpg --mode adaptive --epsilon 0.05 --iterations 30

Увага: у режимах threshold/adaptive початкова фотографія є еталоном
для обчислення похибки, а модель отримує лише вибрані пікселі.
Індексація Python починається з 0; у LaTeX — з 1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator


def read_grayscale(path: Path, max_side: int) -> np.ndarray:
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert('L')
        if max_side > 0 and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        array = np.asarray(im, dtype=np.float64) / 255.0
    if min(array.shape) < 2:
        raise ValueError('Потрібне зображення розміром щонайменше 2x2.')
    return array


def save_grayscale(path: Path, arr: np.ndarray) -> None:
    pixels = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode='L').save(path)


def thomas_algorithm(lower: np.ndarray, diagonal: np.ndarray,
                     upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """A x = rhs, тридіагональна A, O(n)."""
    lower = np.asarray(lower, dtype=float).copy()
    diagonal = np.asarray(diagonal, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float).copy()
    n = len(diagonal)
    if n < 2 or len(lower) != n - 1 or len(upper) != n - 1 or len(rhs) != n:
        raise ValueError('Невірні розміри системи')
    for k in range(1, n):
        q = lower[k - 1] / diagonal[k - 1]
        diagonal[k] -= q * upper[k - 1]
        rhs[k] -= q * rhs[k - 1]
    solution = np.empty(n)
    solution[-1] = rhs[-1] / diagonal[-1]
    for k in range(n - 2, -1, -1):
        solution[k] = (rhs[k] - upper[k] * solution[k + 1]) / diagonal[k]
    return solution


def spline_coefficients_1d(samples: np.ndarray) -> np.ndarray:
    """c[k-1]+4c[k]+c[k+1]=6 f[k] із дзеркальними краями."""
    n = len(samples)
    lower = np.ones(n - 1)
    upper = np.ones(n - 1)
    lower[-1] = 2.0
    upper[0] = 2.0
    return thomas_algorithm(lower, np.full(n, 4.0), upper, 6.0 * samples)


def exact_coefficients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Спершу рядки D, потім стовпці C — точні B-сплайн-коефіцієнти."""
    m, n = image.shape
    d = np.vstack([spline_coefficients_1d(image[i]) for i in range(m)])
    c = np.column_stack([spline_coefficients_1d(d[:, j]) for j in range(n)])
    return d, c


def mirror_indices(indices: np.ndarray, n: int) -> np.ndarray:
    period = 2 * (n - 1)
    t = np.mod(indices, period)
    return np.where(t < n, t, period - t).astype(int)


def beta3(t: np.ndarray) -> np.ndarray:
    a = np.abs(np.asarray(t, dtype=float))
    return np.where(a < 1, 2.0 / 3 - a * a + 0.5 * a**3,
                    np.where(a < 2, (2 - a)**3 / 6, 0.0))


def evaluation_basis(coords: np.ndarray, n: int) -> sparse.csr_matrix:
    """Рядок B: не більше 4 ненульових B-сплайнових ваг."""
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    if n < 2 or coords.ndim != 1 or not np.all(np.isfinite(coords)):
        raise ValueError('Невірні координати для B-сплайна')
    base = np.floor(coords).astype(int)
    shift = np.arange(-1, 3)
    unwrapped = base[:, None] + shift[None, :]
    col = mirror_indices(unwrapped, n)
    weights = beta3(coords[:, None] - unwrapped)
    rows = np.repeat(np.arange(len(coords)), 4)
    return sparse.coo_matrix((weights.ravel(), (rows, col.ravel())),
                             shape=(len(coords), n)).tocsr()


def interpolate_grid(coeff: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S(x,y)=Bx C By^T; еквівалент локальної суми з 16 членів."""
    bx = evaluation_basis(x, coeff.shape[0])
    by = evaluation_basis(y, coeff.shape[1])
    return np.asarray((bx @ coeff) @ by.T.toarray())


def second_difference(n: int) -> sparse.csr_matrix:
    if n <= 2:
        return sparse.csr_matrix((0, n))
    ones = np.ones(n - 2)
    return sparse.diags([ones, -2 * ones, ones], [0, 1, 2],
                        shape=(n - 2, n), format='csr')


class RegularizedSpline:
    """min ||P_Ω H vec(C)-I_Ω||² + λ( ||D²_x C||² + ||C D²_y^T||² + η||C||² ).

    H = kron(B_M, B_N), бо vec(B_M C B_N^T)=H vec(C) у порядку C.
    """
    def __init__(self, image: np.ndarray, lam: float, eta: float):
        if lam <= 0 or eta <= 0:
            raise ValueError('Параметри lambda та eta повинні бути додатними')
        self.image = image
        self.m, self.n = image.shape
        bx = evaluation_basis(np.arange(self.m), self.m)
        by = evaluation_basis(np.arange(self.n), self.n)
        self.h = sparse.kron(bx, by, format='csr')
        dx = second_difference(self.m)
        dy = second_difference(self.n)
        eye_m = sparse.eye(self.m, format='csr')
        eye_n = sparse.eye(self.n, format='csr')
        self.reg = (
            sparse.kron(dx.T @ dx, eye_n, format='csr')
            + sparse.kron(eye_m, dy.T @ dy, format='csr')
            + eta * sparse.eye(self.m * self.n, format='csr')
        )
        self.lam = lam
        self._previous = None

    def fit(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = np.flatnonzero(mask.ravel())
        if ids.size == 0:
            raise ValueError('Множина опорних точок порожня: змініть поріг tau')
        h_known = self.h[ids, :]
        a = (h_known.T @ h_known + self.lam * self.reg).tocsr()
        b = np.asarray(h_known.T @ self.image.ravel()[ids]).ravel()
        diag = a.diagonal()
        preconditioner = LinearOperator(a.shape, matvec=lambda x: x / diag)
        coeff, info = cg(a, b, x0=self._previous, rtol=1e-9, atol=0,
                         maxiter=1800, M=preconditioner)
        if info != 0:
            raise ArithmeticError(f'Спряжені градієнти не збіглися: info={info}. '
                                  'Спробуйте збільшити lambda або eta.')
        self._previous = coeff
        predicted = np.asarray(self.h @ coeff).reshape(self.m, self.n)
        return coeff.reshape(self.m, self.n), predicted


def mse_psnr(original: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((original - predicted)**2))
    return mse, float('inf') if mse == 0 else float(10 * np.log10(1 / mse))


def make_comparison(original: np.ndarray, result: np.ndarray,
                    output: Path, mask: np.ndarray | None = None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5))
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Вхідне зображення')
    axes[1].imshow(result, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Результат B-сплайна')
    if mask is not None:
        axes[2].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Маска опорних пікселів')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_self_test() -> None:
    rng = np.random.default_rng(1234)
    image = rng.random((7, 9))
    d, c = exact_coefficients(image)
    recovered = interpolate_grid(c, np.arange(7), np.arange(9))
    e = float(np.max(np.abs(recovered - image)))
    assert e < 1e-12, f'Exact test failed: {e}'
    assert d.shape == c.shape == image.shape
    model = RegularizedSpline(image, lam=0.05, eta=0.001)
    mask = np.zeros_like(image, dtype=bool)
    mask[::2, ::3] = True
    cs, approx = model.fit(mask)
    assert np.isfinite(cs).all() and np.isfinite(approx).all()
    assert approx.shape == image.shape
    print(f'SELF-TEST OK: E_int={e:.3e}; sparse reconstruction is finite')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, help='Шлях до вихідного зображення')
    parser.add_argument('--mode', choices=['exact', 'threshold', 'adaptive'], default='exact')
    parser.add_argument('--output', type=Path, default=Path('results_algorithm_1'))
    parser.add_argument('--scale', type=float, default=2.0, help='Масштаб збільшення (>=1)')
    parser.add_argument('--max-side', type=int, default=None,
                        help='Максимальна сторона перед обробкою; 0 = без зменшення')
    parser.add_argument('--tau', type=float, default=0.40, help='Поріг відбору, I>tau')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Критерій адаптивної похибки')
    parser.add_argument('--lambda-reg', type=float, default=0.01, help='Вага R(C)')
    parser.add_argument('--eta', type=float, default=0.001, help='Стабілізація у R(C)')
    parser.add_argument('--iterations', type=int, default=30, help='Макс. нових точок adaptive')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.input is None:
        parser.error('Потрібно вказати --input або --self-test')
    if args.scale < 1 or not 0 <= args.tau <= 1 or args.epsilon <= 0 or args.iterations < 0:
        parser.error('Перевірте scale>=1, tau∈[0,1], epsilon>0, iterations>=0')

    # Для неповних даних регуляризована модель істотно дорожча за exact.
    max_side = args.max_side if args.max_side is not None else (
        256 if args.mode == 'exact' else 128 if args.mode == 'threshold' else 64
    )
    image = read_grayscale(args.input, max_side)
    m, n = image.shape
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    save_grayscale(output / 'original.png', image)
    mask = None
    report = [f'Режим: {args.mode}', f'Розмір: {m} x {n}',
              f'Передобробка: max_side={max_side}; значення нормовано [0,1]']

    if args.mode == 'exact':
        intermediate, coeff = exact_coefficients(image)
        np.save(output / 'coefficients_D.npy', intermediate)
        reconstructed = interpolate_grid(coeff, np.arange(m), np.arange(n))
        e_int = float(np.max(np.abs(reconstructed - image)))
        report.append(f'E_int = {e_int:.12g}')
        print(f'Точна інтерполяція: E_int = {e_int:.3e}')
    else:
        model = RegularizedSpline(image, lam=args.lambda_reg, eta=args.eta)
        if args.mode == 'threshold':
            mask = image > args.tau
            if not np.any(mask):
                raise ValueError('Omega_tau порожня: зменште --tau')
            coeff, reconstructed = model.fit(mask)
            report.append(f'tau={args.tau}, lambda={args.lambda_reg}, eta={args.eta}')
        else:
            mask = np.zeros_like(image, dtype=bool)
            mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = True
            steps = 0
            for step in range(args.iterations + 1):
                coeff, reconstructed = model.fit(mask)
                errors = np.abs(reconstructed - image)
                errors[mask] = -np.inf
                max_err = float(np.max(errors))
                if not np.isfinite(max_err) or max_err <= args.epsilon:
                    break
                if step == args.iterations:
                    break
                point = np.unravel_index(np.argmax(errors), image.shape)
                mask[point] = True
                steps += 1
            achieved = max_err <= args.epsilon or np.all(mask)
            report.extend([f'epsilon={args.epsilon}, iterations={steps}',
                           f'Макс. похибка на пропущених = {max_err:.8g}',
                           f'Критерій epsilon досягнуто: {achieved}'])
            print(f'Адаптивний режим: {steps} уточнень, поріг досягнуто = {achieved}')
        report.append(f'Кількість опорних точок = {int(mask.sum())} із {m*n}')
        report.append(f'rho = {mask.mean():.8f}')
        if np.any(~mask):
            report.append(f'MSE_miss = {float(np.mean((np.clip(reconstructed,0,1)[~mask]-image[~mask])**2)):.9g}')
        print(f'Опорних пікселів: {int(mask.sum())}/{m*n} ({100*mask.mean():.2f}%)')
        np.save(output / 'support_mask.npy', mask)
        save_grayscale(output / 'support_mask.png', mask.astype(float))

    np.save(output / 'coefficients_C.npy', coeff)
    result_clipped = np.clip(reconstructed, 0, 1)
    save_grayscale(output / 'reconstructed.png', result_clipped)
    mse, psnr = mse_psnr(image, result_clipped)
    report += [f'MSE = {mse:.12g}', f'PSNR, dB = {psnr:.9g}']

    # Збільшення працює для всіх режимів, а не лише exact.
    rows = int(round((m - 1) * args.scale)) + 1
    cols = int(round((n - 1) * args.scale)) + 1
    enlarged = interpolate_grid(coeff, np.linspace(0, m-1, rows),
                                np.linspace(0, n-1, cols))
    save_grayscale(output / 'interpolated.png', enlarged)
    np.save(output / 'interpolated_raw.npy', enlarged)
    make_comparison(image, result_clipped, output / 'comparison.png', mask)
    report.append(f'Збільшена матриця: {rows} x {cols}, scale={args.scale}')
    (output / 'report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'MSE = {mse:.6g}; PSNR = {psnr:.4f} dB')
    print(f'Результати: {output.resolve()}')


if __name__ == '__main__':
    main()
