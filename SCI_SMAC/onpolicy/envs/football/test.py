import numpy as np
import matplotlib.pyplot as plt


# p(S_{t+1}^{c}|A_{t}^{1})
matrix1 = [[0.3,
            0.3,
            0.4],
           [0.4,
            0.3,
            0.3]]

# p(S_{t+1}^{c}|A_{t}^{2})
matrix2 = [[0.1,
            0.8,
            0.1],
           [0.1,
            0.1,
            0.8]]

def calculate_mi(pa11, pa12, pa21, pa22):
    pa11_s1 = 0.3 * pa11
    pa11_s2 = 0.3 * pa11
    pa11_s3 = 0.4 * pa11

    pa12_s1 = 0.4 * pa12
    pa12_s2 = 0.3 * pa12
    pa12_s3 = 0.3 * pa12

    p1_s1 = pa11_s1 + pa12_s1
    p1_s2 = pa11_s2 + pa12_s2
    p1_s3 = pa11_s3 + pa12_s3

    mi_1 = pa11_s1 * np.log(0.3 / p1_s1) + pa12_s1 * np.log(0.4 / p1_s1) + \
           pa11_s2 * np.log(0.3 / p1_s2) + pa12_s2 * np.log(0.3 / p1_s2) + \
           pa11_s3 * np.log(0.4 / p1_s3) + pa12_s3 * np.log(0.3 / p1_s3)

    pa21_s1 = 0.1 * pa21
    pa21_s2 = 0.8 * pa21
    pa21_s3 = 0.1 * pa21

    pa22_s1 = 0.1 * pa22
    pa22_s2 = 0.1 * pa22
    pa22_s3 = 0.8 * pa22

    p2_s1 = pa21_s1 + pa22_s1
    p2_s2 = pa21_s2 + pa22_s2
    p2_s3 = pa21_s3 + pa22_s3

    mi_2 = pa21_s1 * np.log(0.1 / p2_s1) + pa22_s1 * np.log(0.1 / p2_s1) + \
           pa21_s2 * np.log(0.8 / p2_s2) + pa22_s2 * np.log(0.1 / p2_s2) + \
           pa21_s3 * np.log(0.1 / p2_s3) + pa22_s3 * np.log(0.8 / p2_s3)

    return mi_1, mi_2

# Generate values for pa11 and pa21
pa_values = np.linspace(0, 1, 101)
mi1_values = []
mi2_values = []

# Calculate MI for each value of pa11 and pa21
for pa11 in pa_values:
    pa12 = 1 - pa11
    pa21 = pa11  # Assuming pa11 and pa21 are same for the plot. Adjust as necessary.
    pa22 = 1 - pa21
    mi1, mi2 = calculate_mi(pa11, pa12, pa21, pa22)
    mi1_values.append(mi1)
    mi2_values.append(mi2)

# Plot the results
plt.figure(figsize=(9, 5))
plt.plot(pa_values, mi1_values, label=r'Agent 1', linewidth=3)
plt.plot(pa_values, mi2_values, label=r'Agent 2', linewidth=3)
plt.xlabel(r'$p(a_{1})$', fontsize=35)
plt.ylabel(r'$I(S_{t+1}^{c}; A_{t}^{i})$', fontsize=35)
# plt.title(r'Mutual Information', fontsize=20)
plt.legend(loc='upper right', fontsize=16)
# Set the font size for xticks and yticks
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.tight_layout(pad=0.5)

plt.savefig('credit_metric_validation.svg', format='svg', dpi=600)
# plt.savefig('credit_metric_validation.pdf', format='pdf', dpi=600)

plt.show()