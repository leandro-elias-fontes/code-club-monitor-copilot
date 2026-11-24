"""Deterministic platform & competitive programming bug detectors (Beecrowd, NEPS, Codeforces, AtCoder)."""

from __future__ import annotations

import re

from pydantic import BaseModel

from src.static_analyzer import Language, _strip_cpp_comments_and_strings


class PlatformBugFinding(BaseModel):
    """High-precision deterministic bug detected from platform/language patterns."""

    bug_class: str
    title: str
    bug_location: str
    counterexample: str
    how_to_fix: str
    monitor_tip: str
    suspect_line: int | None = None


def detect_platform_bugs(
    code: str,
    language: Language,
    problem_statement: str = "",
    verdict: str = "unknown",
) -> PlatformBugFinding | None:
    """Detect high-precision pitfalls common in Beecrowd, NEPS, Codeforces, and AtCoder."""
    stmt = (problem_statement or "").lower()
    cleaned = _strip_cpp_comments_and_strings(code) if language in ("cpp", "c") else code
    lines = cleaned.splitlines()

    if language in ("cpp", "c"):
        # 1. Beecrowd/NEPS: missing fixed << setprecision(k) when decimal places are requested
        m_dec = re.search(r"(\d+)\s+casas?\s+decimais|(\d+)\s+decimal\s+places?", stmt)
        if m_dec and "cout" in cleaned and "setprecision" not in cleaned:
            places = m_dec.group(1) or m_dec.group(2)
            line_num = next((i for i, ln in enumerate(lines, 1) if "cout" in ln), None)
            return PlatformBugFinding(
                bug_class="presentation_or_precision",
                title=f"Formatação Decimal Ausente (faltou fixed << setprecision({places}))",
                bug_location=(
                    f"O enunciado exige exatamente {places} casas decimais na saída, mas o código usa 'cout' sem "
                    f"'fixed << setprecision({places})'. No Beecrowd/NEPS isso causa Wrong Answer ou Presentation Error."
                ),
                counterexample=(
                    f"Valor calculado: 12.5\n"
                    f"Saída atual do cout: 12.5\n"
                    f"Saída exigida ({places} casas): 12.{'5' + '0' * (int(places) - 1)}"
                ),
                how_to_fix=(
                    f"Inclua '#include <iomanip>' e configure 'cout << fixed << setprecision({places});' antes de imprimir."
                ),
                monitor_tip=(
                    f"Pergunte ao aluno: 'O enunciado pede {places} casas decimais sempre — "
                    "o que o cout imprime quando o resultado termina em .50 ou .00?'"
                ),
                suspect_line=line_num,
            )

        # 2. C/C++ integer division truncation when decimal/average is expected
        expects_decimal = bool(
            m_dec or re.search(r"\b(media|média|average|porcentagem|percentual|real|ponto flutuante)\b", stmt)
        )
        if expects_decimal and "double" not in cleaned and "float" not in cleaned:
            for idx, ln in enumerate(lines, 1):
                if re.search(r"(?:\)|[a-zA-Z_]\w*)\s*/\s*(?:\(|[a-zA-Z_]\w*|\d+)\b", ln) and ".0" not in ln:
                    return PlatformBugFinding(
                        bug_class="presentation_or_precision",
                        title="Divisão Inteira Truncada em C/C++ (int / int)",
                        bug_location=(
                            f"Linha {idx}: A questão espera um valor decimal, mas as variáveis e a divisão usam apenas 'int'. "
                            "Em C/C++, dividir dois inteiros descarta a parte decimal (ex: 5 / 2 vira 2 em vez de 2.5)."
                        ),
                        counterexample=(
                            "Entrada: 5 e 2\n"
                            "Saída do Aluno (int / int): 2\n"
                            "Saída Esperada (double): 2.50"
                        ),
                        how_to_fix=(
                            f"Na linha {idx}, use tipo 'double' nas variáveis ou force conversão com '1.0 * ...' / '(double)...'."
                        ),
                        monitor_tip=(
                            "Pergunte ao aluno: 'Em C/C++, quanto dá 5 / 2 quando os dois números são int? "
                            "Como fazemos para o computador não jogar fora as casas decimais?'"
                        ),
                        suspect_line=idx,
                    )

        # 3. cin >> followed by getline without cin.ignore()
        saw_cin = False
        for idx, ln in enumerate(lines, 1):
            if re.search(r"\bcin\s*>>", ln):
                saw_cin = True
            if "cin.ignore" in ln or "ws" in ln:
                saw_cin = False
            if saw_cin and re.search(r"\bgetline\s*\(", ln):
                return PlatformBugFinding(
                    bug_class="edge_case_uninitialized_or_div0",
                    title="Leitura de Linha Vazia após cin >> (faltou cin.ignore())",
                    bug_location=(
                        f"Linha {idx}: O código usa 'cin >>' e depois chama 'getline(...)'. O 'cin >>' deixa o '\\n' "
                        "no buffer, fazendo o primeiro 'getline' ler uma string vazia!"
                    ),
                    counterexample=(
                        "Entrada:\n1\nTexto da linha\n"
                        "Leitura do Aluno: s = \"\" (string vazia)\n"
                        "Leitura Esperada: s = \"Texto da linha\""
                    ),
                    how_to_fix=f"Adicione 'cin.ignore();' logo antes do 'getline' na linha {idx} (ou use 'getline(cin >> ws, s)').",
                    monitor_tip=(
                        "Pergunte ao aluno: 'Quando o usuário aperta Enter após digitar o número no cin >>, "
                        "onde vai parar o caractere de quebra de linha antes do getline rodar?'"
                    ),
                    suspect_line=idx,
                )

        # 4. 32-bit bit-shift overflow: 1 << k instead of 1LL << k
        for idx, ln in enumerate(lines, 1):
            if re.search(r"\b1\s*<<\s*[a-zA-Z_]\w*", ln) and "1LL" not in ln and "1ll" not in ln:
                if "long long" in cleaned or "30" in stmt or "60" in stmt or "10^9" in stmt:
                    return PlatformBugFinding(
                        bug_class="integer_overflow",
                        title="Overflow em Bit-Shift 32-bit (1 << k em vez de 1LL << k)",
                        bug_location=(
                            f"Linha {idx}: A expressão '1 << k' usa o literal '1' de 32 bits. "
                            "Para k >= 31 ocorre Undefined Behavior / Integer Overflow mesmo guardando em 'long long'."
                        ),
                        counterexample=(
                            "Entrada: k = 40\n"
                            "Saída com (1 << 40): Undefined Behavior / valor negativo ou 256\n"
                            "Saída Esperada (1LL << 40): 1099511627776"
                        ),
                        how_to_fix=f"Na linha {idx}, troque '1 <<' por '1LL <<' para forçar deslocamento em 64 bits.",
                        monitor_tip=(
                            f"Pergunte ao aluno: 'Na linha {idx}, o número 1 antes do << tem quantos bits? "
                            "O que acontece se deslocarmos ele mais de 31 casas para a esquerda?'"
                        ),
                        suspect_line=idx,
                    )

        # 5. memset with invalid non-byte value (e.g., memset(dp, 1, sizeof(dp)))
        for idx, ln in enumerate(lines, 1):
            m_mem = re.search(r"\bmemset\s*\(\s*[^,]+,\s*([^,]+),", ln)
            if m_mem:
                val = m_mem.group(1).strip()
                if val not in ("0", "-1", "0x3f", "0x7f", "0xff", "false", "true"):
                    return PlatformBugFinding(
                        bug_class="edge_case_uninitialized_or_div0",
                        title=f"Uso Incorreto de memset com valor '{val}'",
                        bug_location=(
                            f"Linha {idx}: 'memset(..., {val}, ...)' preenche cada BYTE individualmente, não cada inteiro. "
                            "Para inteiros, 'memset' só funciona corretamente com 0, -1 ou 0x3f."
                        ),
                        counterexample=(
                            f"Chamada: memset(v, {val}, sizeof(v))\n"
                            "Valor real em cada int: 0x01010101 (16.843.009) em vez de 1!"
                        ),
                        how_to_fix="Substitua 'memset' por um loop 'for' ou 'fill(v, v + n, valor);'.",
                        monitor_tip=(
                            f"Pergunte ao aluno: 'Você sabe se o memset preenche cada posição do vetor ou cada byte da memória? "
                            f"Qual valor fica dentro de um int de 4 bytes se colocarmos {val} em cada byte?'"
                        ),
                        suspect_line=idx,
                    )

        # 6. Negative modulo in C/C++: (a - b) % MOD without + MOD
        for idx, ln in enumerate(lines, 1):
            if re.search(r"\([^)]*-[^)]*\)\s*%\s*(?:mod|MOD|1000000007|998244353)", ln) and "+ mod" not in ln.lower():
                return PlatformBugFinding(
                    bug_class="edge_case_uninitialized_or_div0",
                    title="Módulo Negativo em Subtração C/C++",
                    bug_location=(
                        f"Linha {idx}: Em C/C++, o operador '%' preserva o sinal negativo (ex: (-3) % 10 == -3). "
                        "Ao subtrair antes do módulo, a resposta pode ficar negativa."
                    ),
                    counterexample=(
                        "Exemplo: a = 2, b = 5, MOD = 1000000007\n"
                        "Saída em C++ ((2 - 5) % MOD): -3\n"
                        "Saída Esperada: 1000000004"
                    ),
                    how_to_fix=f"Na linha {idx}, normalize o resto somando MOD: '((a - b) % MOD + MOD) % MOD'.",
                    monitor_tip=(
                        "Pergunte ao aluno: 'Em C++, quanto dá (-3) % 7? O resultado sai positivo como na matemática ou negativo?'"
                    ),
                    suspect_line=idx,
                )

    else:
        # Python-specific high-precision pitfalls
        # 1. Shallow 2D matrix copy: [[0] * m] * n
        for idx, ln in enumerate(lines, 1):
            if re.search(r"\[\s*\[[^\]]+\]\s*\*\s*[^\]\n]+\]\s*\*\s*[^\]\n]+", ln):
                return PlatformBugFinding(
                    bug_class="edge_case_uninitialized_or_div0",
                    title="Cópia Rasa de Matriz 2D em Python ([[0] * M] * N)",
                    bug_location=(
                        f"Linha {idx}: Multiplicar uma lista de listas em Python cria N referências para a MESMA linha na memória. "
                        "Alterar matriz[0][j] altera todas as linhas matriz[i][j] ao mesmo tempo!"
                    ),
                    counterexample=(
                        "Código: mat = [[0]*3]*2; mat[0][1] = 9\n"
                        "Resultado obtido: [[0, 9, 0], [0, 9, 0]]\n"
                        "Resultado esperado: [[0, 9, 0], [0, 0, 0]]"
                    ),
                    how_to_fix=f"Na linha {idx}, crie linhas independentes com list comprehension: '[[0] * m for _ in range(n)]'.",
                    monitor_tip=(
                        f"Pergunte ao aluno: 'Na linha {idx}, se você mudar mat[0][0] = 1 e imprimir mat[1][0], o que aparece?'"
                    ),
                    suspect_line=idx,
                )

        # 2. Recursive DFS/function without sys.setrecursionlimit for large N
        if "def " in cleaned and "setrecursionlimit" not in cleaned and re.search(r"\b(10\^?[456]|100000)\b", stmt):
            for idx, ln in enumerate(lines, 1):
                m_def = re.match(r"\s*def\s+([a-zA-Z_]\w*)\s*\(", ln)
                if m_def:
                    fname = m_def.group(1)
                    if any(re.search(rf"\b{fname}\s*\(", l2) for l2 in lines[idx:]):
                        return PlatformBugFinding(
                            bug_class="infinite_loop_or_recursion",
                            title="Limite de Recursão Padrão do Python (1000 chamadas)",
                            bug_location=(
                                f"Linha {idx}: A função recursiva '{fname}' pode exceder o limite padrão de 1000 "
                                "chamadas do Python para N >= 10^4, causando RecursionError (RTE)."
                            ),
                            counterexample=(
                                "Entrada: Grafo em linha com N = 2000 vértices\n"
                                "Saída do Aluno: RecursionError: maximum recursion depth exceeded\n"
                                "Saída Esperada: Execução normal ou versão iterativa"
                            ),
                            how_to_fix="Adicione 'import sys; sys.setrecursionlimit(200000)' ou converta a travessia para iterativa (pilha/fila).",
                            monitor_tip=(
                                "Pergunte ao aluno: 'Quantas chamadas recursivas seguidas o Python aceita por padrão antes de dar RecursionError?'"
                            ),
                            suspect_line=idx,
                        )

    # Presentation Error hint when monitor explicitly selected PE
    if verdict == "PE":
        return PlatformBugFinding(
            bug_class="presentation_or_precision",
            title="Presentation Error (Espaço Extra ou Quebra de Linha)",
            bug_location=(
                "O juiz retornou Presentation Error (PE): os valores estão corretos, mas há espaço em branco sobrando "
                "no final da linha, falta de '\\n' no último print ou diferença de maiúsculas/minúsculas."
            ),
            counterexample=(
                "Saída comum com erro: '1 2 3 ' (espaço antes do \\n) ou sem quebra de linha final\n"
                "Saída exigida (Beecrowd/NEPS): '1 2 3\\n'"
            ),
            how_to_fix=(
                "Imprima o separador ' ' apenas entre elementos (ex: `if (i > 0) cout << ' ';`) e garanta `\\n` ao final."
            ),
            monitor_tip=(
                "Pergunte ao aluno: 'O seu loop está imprimindo um espaço depois do último elemento ou faltou uma quebra de linha no final?'"
            ),
            suspect_line=None,
        )

    return None
