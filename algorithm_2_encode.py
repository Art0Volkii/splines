#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""АЛГОРИТМ 2. Адаптивне сплайнове КОДУВАННЯ зображення у градаціях сірого.

Ідейна основа: [3]. Конкретна чисельна реалізація (уточнена в роботі):
  1) послідовно по рядках, а потім по стовпцях формувати природні
     кубічні сплайни з адаптивним вибором опорних точок;
  2) якщо максимальна похибка > відповідного порога, додати піксель
     найбільшої похибки і його сусіда, перебудувати поліноми;
  3) пороги двох проходів = epsilon/2, тому |I-X| <= epsilon;
  4) зберегти лише остаточні СТОВПЦЕВІ кубічні сегменти, які повністю
     визначають X: чотири коефіцієнти кожного полінома і його межі;
     опорні значення задані безпосередньо значеннями сегментів у вузлах;
     проміжні рядкові коефіцієнти не потрібні для декодування;
  5) квантована різниця q = round((I-X)/delta);
  6) контейнер npz: всі метадані та масиви ZIP/Deflate-стиснені.

Відновлення X за збереженими поліномами НЕ потребує повторного
обчислення коефіцієнтів сплайнів. Інший скрипт: algorithm_3_decode.py.

Наявність квантування і ZIP-стиснення не гарантує коефіцієнт
стиснення понад одиницю. У звіті враховується ВЕСЬ розмір .npz.

Встановлення: python -m pip install numpy scipy pillow matplotlib
Приклад:
  python algorithm_2_encode.py --input photo.png --output coded.npz --epsilon 0.025 --delta 0.01
  python algorithm_2_encode.py --input photo.png --max-side 128 --output coded.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.interpolate import CubicSpline


def load_image(path: Path, max_side: int = 0) -> np.ndarray:
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert('L')
        if max_side > 0 and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image = np.asarray(im, dtype=np.float64) / 255.0
    if min(image.shape) < 2:
        raise ValueError('Розмір зображення повинен бути не меншим за 2x2')
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(pixels, mode='L').save(path)


def adaptive_spline(samples: np.ndarray, tolerance: float):
    """Повертає (вузли, CubicSpline, наближення) із похибкою <= tolerance.

    За великої похибки додає пару сусідніх пікселів; повторює,
    поки всі значення не відповідають допуску. Збіжність скінченна:
    якщо всі пікселі стануть вузлами, помилка у вузлах ~ 1e-15.
    """
    f = np.asarray(samples, dtype=np.float64)
    n = len(f)
    if n < 2 or tolerance <= 0:
        raise ValueError('Потрібно n>=2 і tolerance>0')
    initial = min(4, n)
    anchors = set(np.rint(np.linspace(0, n - 1, initial)).astype(int).tolist())
    x = np.arange(n, dtype=np.float64)
    # Математично кероване уточнення: не більше n додавань.
    for _ in range(n + 1):
        knots = np.asarray(sorted(anchors), dtype=np.int32)
        cs = CubicSpline(knots.astype(float), f[knots], bc_type='natural')
        approximated = cs(x)
        error = np.abs(f - approximated)
        worst = int(np.argmax(error))
        if error[worst] <= tolerance + 1e-12:
            return knots, cs, approximated
        if len(anchors) == n:
            return knots, cs, approximated
        # Уточнення для точки та сусідньої: правила на межі зафіксовані.
        anchors.add(worst)
        right = worst + 1 if worst + 1 < n else worst - 1
        anchors.add(right)
    raise ArithmeticError('Не вдалося досягти допустимої похибки')


def encode(image: np.ndarray, epsilon: float, delta: float):
    """Стискання, що зберігає кубічні коефіцієнти КІНЦЕВОГО наближення X."""
    if epsilon <= 0 or delta <= 0 or not np.isfinite([epsilon, delta]).all():
        raise ValueError('epsilon та delta повинні бути скінченними додатними')
    m, n = image.shape
    row_tol = epsilon / 2
    col_tol = epsilon / 2

    # 1. Наближення за рядками.
    row_image = np.empty_like(image)
    row_knot_count = np.zeros(m, dtype=np.int32)
    for i in range(m):
        knots, _cs, row_image[i] = adaptive_spline(image[i], row_tol)
        row_knot_count[i] = len(knots)

    # 2. Наближення за стовпцями. ЗБЕРІГАЄМО коефіцієнти кожного сегмента.
    x_approx = np.empty_like(image)
    knot_offsets = [0]
    segment_offsets = [0]
    all_knots = []
    all_coefficients = []
    col_knot_count = np.zeros(n, dtype=np.int32)
    for j in range(n):
        knots, cs, x_approx[:, j] = adaptive_spline(row_image[:, j], col_tol)
        col_knot_count[j] = len(knots)
        all_knots.extend(knots.tolist())
        # CubicSpline.c shape (4, knots-1):
        # S(x) = a*(x-k)^3 + b*(x-k)^2 + c*(x-k) + d
        all_coefficients.extend(cs.c.T.tolist())
        knot_offsets.append(len(all_knots))
        segment_offsets.append(len(all_coefficients))

    approx_error = float(np.max(np.abs(image - x_approx)))
    if approx_error > epsilon + 1e-9:
        raise ArithmeticError(f'Не виконано оцінку |I-X|<=epsilon: {approx_error}')

    # 3. Різницева складова та рівномірне скалярне квантування.
    residual = image - x_approx
    quantized_float = np.rint(residual / delta)
    max_abs_q = np.max(np.abs(quantized_float))
    if max_abs_q > np.iinfo(np.int32).max:
        raise ValueError('Крок delta надто малий: квантовані індекси переповнюють int32')
    q_dtype = np.int16 if max_abs_q <= np.iinfo(np.int16).max else np.int32
    q = quantized_float.astype(q_dtype)
    reconstructed = np.clip(x_approx + delta * q, 0.0, 1.0)

    data = {
        'format_version': np.asarray(1, dtype=np.int32),
        'image_shape': np.asarray([m, n], dtype=np.int32),
        'epsilon': np.asarray(epsilon, dtype=np.float64),
        'delta': np.asarray(delta, dtype=np.float64),
        'knot_offsets': np.asarray(knot_offsets, dtype=np.int32),
        'segment_offsets': np.asarray(segment_offsets, dtype=np.int32),
        'support_indices': np.asarray(all_knots, dtype=np.int32),
        'segment_coefficients': np.asarray(all_coefficients, dtype=np.float64).reshape(-1, 4),
        'residual_quantized': q,
        'row_support_count': row_knot_count,
        'column_support_count': col_knot_count,
    }
    stats = {
        'max_approx_error': approx_error,
        'max_quantization_error': float(np.max(np.abs(residual - delta * q))),
        'mse_reconstructed': float(np.mean((image - reconstructed)**2)),
        'row_support_total': int(np.sum(row_knot_count)),
        'column_support_total': int(np.sum(col_knot_count)),
    }
    return data, stats, x_approx, reconstructed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('encoded_spline.npz'))
    parser.add_argument('--epsilon', type=float, default=0.025,
                        help='Допустима абсолютна похибка низькочастотного наближення')
    parser.add_argument('--delta', type=float, default=0.01,
                        help='Крок квантування різницевої складової')
    parser.add_argument('--max-side', type=int, default=0,
                        help='0=оригінальний розмір; N=примусове зменшення перед кодуванням')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.npz':
        parser.error('Вкажіть файл із розширенням .npz')
    image = load_image(args.input, args.max_side)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data, stats, approximation, reconstructed = encode(image, args.epsilon, args.delta)
    np.savez_compressed(args.output, **data)

    # Файли-додатки НЕ входять до кодованого контейнера.
    save_image(args.output.parent / 'original_used.png', image)
    save_image(args.output.parent / 'approximation_preview.png', approximation)
    save_image(args.output.parent / 'encoder_reconstruction_preview.png', reconstructed)
    n_bytes = image.size  # Для 8-бітного grayscale без метаданих: 1 байт/піксель
    packed = args.output.stat().st_size
    ratio = n_bytes / packed
    report = [
        'АЛГОРИТМ 2 — КОДУВАННЯ',
        f'Розмір обробленого зображення: {image.shape[0]} x {image.shape[1]}',
        f'epsilon: {args.epsilon}; delta: {args.delta}',
        f'Кількість опорних вузлів, рядковий етап: {stats["row_support_total"]}',
        f'Кількість опорних вузлів, стовпцевий етап: {stats["column_support_total"]}',
        f'Максимальна похибка наближення |I-X|: {stats["max_approx_error"]:.12g}',
        f'Максимальна похибка квантування: {stats["max_quantization_error"]:.12g}',
        f'MSE відновлення до переведення у uint8: {stats["mse_reconstructed"]:.12g}',
        f'B_orig: {n_bytes} байт (нестиснена матриця uint8)',
        f'B_coded: {packed} байт (повний файл .npz)',
        f'K_comp=B_orig/B_coded: {ratio:.6f}',
        'Увага: якщо K_comp<1, файл ЗБІЛЬШИВСЯ, незважаючи на квантування.',
    ]
    (args.output.parent / 'report_encoding.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))
    print(f'Контейнер: {args.output.resolve()}')


if __name__ == '__main__':
    main()
