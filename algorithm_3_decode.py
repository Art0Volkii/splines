#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""АЛГОРИТМ 3. ДЕКОДУВАННЯ цифрового зображення, закодованого алгоритмом 2.

Вхід: контейнер .npz із збереженими чотирма коефіцієнтами кожного
кубічного полінома, межами ділянок і квантованою різницевою складовою.

  X_hat — обчислення значень з уже збережених коефіцієнтів (без нової
           побудови CubicSpline);
  delta_hat = delta * q;
  I_hat = clip(X_hat + delta_hat, 0, 1).

Встановлення: python -m pip install numpy pillow
Приклад:
  python algorithm_3_decode.py --input encoded_spline.npz --output decoded.png
  python algorithm_3_decode.py --input encoded_spline.npz --reference original_used.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def restore_approximation(payload) -> np.ndarray:
    """Відновлює X ТІЛЬКИ зі збережених коефіцієнтів поліномів."""
    version = int(payload['format_version'])
    if version != 1:
        raise ValueError(f'Непідтримувана версія контейнера: {version}')
    shape = np.asarray(payload['image_shape'], dtype=int)
    if shape.shape != (2,) or min(shape) < 2:
        raise ValueError('Неправильний image_shape')
    m, n = map(int, shape)
    ko = np.asarray(payload['knot_offsets'], dtype=int)
    so = np.asarray(payload['segment_offsets'], dtype=int)
    knots_all = np.asarray(payload['support_indices'], dtype=int)
    coeffs_all = np.asarray(payload['segment_coefficients'], dtype=float)
    if (ko.shape != (n + 1,) or so.shape != (n + 1,)
        or ko[0] != 0 or so[0] != 0
        or ko[-1] != len(knots_all) or so[-1] != len(coeffs_all)
        or coeffs_all.ndim != 2
        or coeffs_all.shape[1] != 4):
        raise ValueError('Некоректні масиви сплайнових сегментів')

    result = np.empty((m, n), dtype=np.float64)
    xs = np.arange(m, dtype=np.float64)
    for j in range(n):
        knots = knots_all[ko[j]:ko[j+1]]
        coefficients = coeffs_all[so[j]:so[j+1]]
        if (len(knots) < 2 or len(coefficients) != len(knots)-1
            or knots[0] != 0 or knots[-1] != m-1
            or np.any(np.diff(knots) <= 0)):
            raise ValueError(f'Некоректні межі поліномів для стовпця {j}')
        # Знайти активну ділянку кожного рядка, потім схема Горнера.
        segment = np.clip(np.searchsorted(knots, xs, side='right')-1,
                          0, len(knots)-2)
        delta_x = xs-knots[segment]
        a, b, c, d = coefficients[segment].T
        result[:, j] = ((a * delta_x + b) * delta_x + c) * delta_x + d
    return result


def decode(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Повертає (I_hat, X_hat, Delta) для звичайного grayscale [0,1]."""
    with np.load(path, allow_pickle=False) as payload:
        approximation = restore_approximation(payload)
        q = np.asarray(payload['residual_quantized'])
        delta = float(payload['delta'])
        if q.shape != approximation.shape or not np.issubdtype(q.dtype, np.integer):
            raise ValueError('Розмір або тип квантованої різниці некоректний')
        if delta <= 0 or not np.isfinite(delta):
            raise ValueError('Крок квантування повинен бути додатним')
        reconstructed = np.clip(approximation + delta * q, 0, 1)
    return reconstructed, approximation, delta


def save_image(path: Path, image: np.ndarray) -> None:
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(pixels, mode='L').save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='Файл encoded_spline.npz')
    parser.add_argument('--output', type=Path, default=Path('decoded.png'))
    parser.add_argument('--reference', type=Path,
                        help='Оригінал того самого розміру, для MSE/PSNR; НЕ використовується для декодування')
    args = parser.parse_args()
    reconstructed, approximation, delta = decode(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(args.output, reconstructed)
    print(f'Зображення відновлено: {args.output.resolve()}')
    print(f'Розмір: {reconstructed.shape[0]} x {reconstructed.shape[1]}; delta={delta}')
    if args.reference is not None:
        with Image.open(args.reference) as source:
            ref = np.asarray(ImageOps.exif_transpose(source).convert('L'), dtype=float)/255
        if ref.shape != reconstructed.shape:
            raise ValueError('Розмір еталона відрізняється від декодованого. '
                             'Використайте original_used.png з папки алгоритму 2.')
        mse = float(np.mean((ref-reconstructed)**2))
        psnr = float('inf') if mse == 0 else 10*np.log10(1/mse)
        max_error = float(np.max(np.abs(ref-reconstructed)))
        report = ['АЛГОРИТМ 3 — ДЕКОДУВАННЯ',
                  f'Розмір: {ref.shape[0]} x {ref.shape[1]}',
                  f'MSE: {mse:.12g}', f'PSNR: {psnr:.9g} dB',
                  f'Максимальна абсолютна похибка у нормованих значеннях: {max_error:.12g}']
        (args.output.parent / 'report_decoding.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
        print('\n'.join(report))


if __name__ == '__main__':
    main()
