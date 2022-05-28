def solution(S):
    for i in range(len(S)):
        if S[i] == "1":
            S = S[i::]
            break
    return S.count('1') + len(S) - 1

s = "10"
print(solution(s))

