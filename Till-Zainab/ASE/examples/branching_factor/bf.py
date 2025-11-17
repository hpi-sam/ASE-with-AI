# 1
def f1(x1):
    y = 1
    if isinstance(x1, int):
        y *= 2
    return y

# 2
def f2(x1):
    y = 1
    if isinstance(x1, int):
        y *= 2
        if x1 % 2 == 0:
            y += 3
    return y

# 3
def f3(x1):
    y = 1
    if isinstance(x1, int):
        y *= 2
        if x1 % 2 == 0:
            y += 3
            if 10 <= x1 < 100:
                y -= 5
    return y

# 4
def f4(x1, x2):
    y = 1
    if isinstance(x1, int):
        y *= 2
        if x1 % 2 == 0:
            y += 3
            if 10 <= x1 < 100:
                y -= 5
                if isinstance(x2, str) and len(x2) == 4:
                    y = y ** 2
    return y

# 5
def f5(x1, x2):
    y = 1
    if isinstance(x1, int):
        y *= 2
        if x1 % 2 == 0:
            y += 3
            if 10 <= x1 < 100:
                y -= 5
                if isinstance(x2, str) and len(x2) == 4:
                    y = y ** 2
                    if (x1 % 10) in (2, 4):
                        y //= 2
    return y

# 6
def f6(x1, x2):
    y = 1
    if isinstance(x1, int):
        y *= 2
        if x1 % 2 == 0:
            y += 3
            if 10 <= x1 < 100:
                y -= 5
                if isinstance(x2, str) and len(x2) == 4:
                    y = y ** 2
                    if (x1 % 10) in (2, 4):
                        y //= 2
                        if x2[0].lower() in {'a', 'e'}:
                            y += 7
    return y

# 7
def f7(x1, x2, x3):
    y = 1
    if isinstance(x1, int):
        y *= 2
        if x1 % 2 == 0:
            y += 3
            if 10 <= x1 < 100:
                y -= 5
                if isinstance(x2, str) and len(x2) == 4:
                    y = y ** 2
                    if (x1 % 10) in (2, 4):
                        y //= 2
                        if x2[0].lower() in {'a', 'e'}:
                            y += 7
                            if isinstance(x3, list) and len(x3) == 3:
                                y *= -1
    return y

# 8
def f8(x1, x2, x3):
    y = 1
    if isinstance(x1, int):
        y *= 2
        if x1 % 2 == 0:
            y += 3
            if 10 <= x1 < 100:
                y -= 5
                if isinstance(x2, str) and len(x2) == 4:
                    y = y ** 2
                    if (x1 % 10) in (2, 4):
                        y //= 2
                        if x2[0].lower() in {'a', 'e'}:
                            y += 7
                            if isinstance(x3, list) and len(x3) == 3:
                                y *= -1
                                if sum(v for v in x3 if isinstance(v, int)) == x1:
                                    y -= 9
    return y

# 9
def f9(x1, x2, x3):
    y = 1
    if isinstance(x1, int):
        y *= 2
        if x1 % 2 == 0:
            y += 3
            if 10 <= x1 < 100:
                y -= 5
                if isinstance(x2, str) and len(x2) == 4:
                    y = y ** 2
                    if (x1 % 10) in (2, 4):
                        y //= 2
                        if x2[0].lower() in {'a', 'e'}:
                            y += 7
                            if isinstance(x3, list) and len(x3) == 3:
                                y *= -1
                                if sum(v for v in x3 if isinstance(v, int)) == x1:
                                    y -= 9
                                    if x2[-1].isdigit():
                                        y = 42
    return y

# 10
def f10(x1, x2, x3):
    y = 1
    if isinstance(x1, int):
        y *= 2
        if x1 % 2 == 0:
            y += 3
            if 10 <= x1 < 100:
                y -= 5
                if isinstance(x2, str) and len(x2) == 4:
                    y = y ** 2
                    if (x1 % 10) in (2, 4):
                        y //= 2
                        if x2[0].lower() in {'a', 'e'}:
                            y += 7
                            if isinstance(x3, list) and len(x3) == 3:
                                y *= -1
                                if sum(v for v in x3 if isinstance(v, int)) == x1:
                                    y -= 9
                                    if x2[-1].isdigit():
                                        y = 42
                                        if x1 % 3 == 0:
                                            y += 100
    return y



# 3 Difficult functions:

def f10(x):
    def sum_digits(n: int) -> int:
        return sum(ord(ch)-48 for ch in str(abs(n)))

    def is_prime(n: int) -> bool:
        if n < 2: return False
        small = [2,3,5,7,11,13,17,19,23,29,31]
        for p in small:
            if n == p: return True
            if n % p == 0: return n == p
        i = 37
        while i * i <= n:
            if n % i == 0: return False
            i += 2
        return True

    def is_triangular(t: int) -> bool:
        import math
        s = int(math.isqrt(8*t + 1))
        return s*s == 8*t + 1

    y = 1
    if isinstance(x, int) and x > 0:
        y *= 2
        s = str(x)
        if len(s) == 18 and s.isdigit():
            y += 3
            a = int(s[:6])
            b = int(s[6:12])
            c = int(s[12:])
            if 100_000 <= a < 1_000_000 and is_prime(a):
                y -= 5
                if b == sum_digits(a) and is_triangular(b):
                    y = y ** 2
                    if a % 10 in (1, 3, 7, 9):
                        y //= 2
                        if s[0] in {'2', '3', '5', '7'}:
                            y += 7
                            if c == (a * b) % 1_000_000:
                                y *= -1
                                if int(str(a)[::-1]) % 3 == 2:
                                    y -= 9
                                    if ((a * b) & 1) == 1 and (c & 1) == 1:
                                        y = 42
                                        if (a * b) % 9 == 0:
                                            y += 100
    return y


def f10s(x):
    import unicodedata, base64, zlib, re, string, math

    y = 1
    if isinstance(x, str):
        y *= 2
        s = x
        n = unicodedata.normalize("NFC", s)
        if s == n and 8 <= len(s) <= 64 and s.isprintable():
            y += 3
            parts = s.split(".")
            if len(parts) == 3:
                y -= 5
                a, b, c = parts
                if len(a) == 6 and a.isalnum() and any(ch.islower() for ch in a) and any(ch.isupper() for ch in a) and sum(ch.isdigit() for ch in a) >= 1:
                    y = y ** 2
                    if a[0].isalpha():
                        y //= 2
                        try:
                            m = base64.b64decode(b, validate=True)
                            t = m.decode("ascii")
                            cleaned = re.sub(r"[^a-z]", "", t.lower())
                            if cleaned and cleaned == cleaned[::-1] and len(t) % 2 == 1:
                                y += 7
                                if len(c) == 8 and all(ch in string.hexdigits for ch in c):
                                    if zlib.crc32(a.encode() + m) % (1 << 32) == int(c, 16):
                                        y *= -1
                                        if (sum(ord(ch) for ch in a) % 8) == int(c[-1], 16) % 8:
                                            y -= 9
                                            digits = "".join(ch for ch in s if ch.isdigit())
                                            if digits and int(digits) % 5 == 0:
                                                y = 42
                                                if (sum(int(ch) for ch in a if ch.isdigit()) * len(cleaned)) % 4 == 0:
                                                    y += 100
                        except Exception:
                            pass
    return y


def f10_list(x):
    y = 1
    if isinstance(x, list):
        y *= 2
        if len(x) == 12:
            y += 3
            if all(isinstance(x[i], str) for i in range(0, 12, 2)) and all(isinstance(x[i], int) for i in range(1, 12, 2)):
                y -= 5
                ints = [x[i] for i in range(1, 12, 2)]
                strs = [x[i] for i in range(0, 12, 2)]
                if all(isinstance(n, int) and -10**6 < n < 10**6 for n in ints) and all(s.isascii() and s.isalpha() for s in strs):
                    y = y ** 2
                    if all(ints[i] < ints[i+1] for i in range(len(ints)-1)) and (sum(ints) % 97 == 0):
                        y //= 2
                        s = "".join(strs)
                        tri_ok = False
                        k = 1
                        while k * (k + 1) // 2 <= len(s):
                            if k * (k + 1) // 2 == len(s):
                                tri_ok = True
                                break
                            k += 1
                        if tri_ok and s.islower():
                            y += 7
                            firsts = [t[0] for t in strs]
                            if len(set(firsts)) == 1 and firsts[0] in "aeiou":
                                y *= -1
                                pal_count = sum(1 for t in strs if t == t[::-1] and len(t) >= 3)
                                if pal_count == 1:
                                    y -= 9
                                    L = [len(x[i]) for i in (2, 4, 6, 8)]
                                    if L[1] - L[0] == L[2] - L[1] == L[3] - L[2] != 0:
                                        y = 42
                                        if ints[-1] == (sum(ints[:-1]) % 1000):
                                            y += 100
    return y


# average and std across difficult functions (int, str, list)