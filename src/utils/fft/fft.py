import torch
import cv2
import numpy as np

try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft


    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))


    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))


    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)


    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)


    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(x, norm=None):
    X1 = idct(x.transpose(-1, -2), norm=norm)
    X2 = idct(X1.transpose(-1, -2), norm=norm)
    return X2


def dct_2d_v2(x, norm=None):
    DCT_MAT = torch.Tensor([
        [0.3535533905932738, 0.4903926402016152, 0.4619397662556434, 0.4157348061512726, 0.3535533905932738,
         0.2777851165098011, 0.1913417161825449, 0.0975451610080642],
        [0.3535533905932738, 0.4157348061512726, 0.1913417161825449, -0.0975451610080641, -0.3535533905932737,
         -0.4903926402016152, -0.4619397662556434, -0.2777851165098011],
        [0.3535533905932738, 0.2777851165098011, -0.1913417161825449, -0.4903926402016152, -0.3535533905932738,
         0.0975451610080642, 0.4619397662556433, 0.4157348061512727],
        [0.3535533905932738, 0.0975451610080642, -0.4619397662556434, -0.2777851165098011, 0.3535533905932737,
         0.4157348061512727, -0.1913417161825450, -0.4903926402016153],
        [0.3535533905932738, -0.0975451610080641, -0.4619397662556434, 0.2777851165098009, 0.3535533905932738,
         -0.4157348061512726, -0.1913417161825453, 0.4903926402016152],
        [0.3535533905932738, -0.2777851165098010, -0.1913417161825452, 0.4903926402016153, -0.3535533905932733,
         -0.0975451610080649, 0.4619397662556437, -0.4157348061512720],
        [0.3535533905932738, -0.4157348061512727, 0.1913417161825450, 0.0975451610080640, -0.3535533905932736,
         0.4903926402016152, -0.4619397662556435, 0.2777851165098022],
        [0.3535533905932738, -0.4903926402016152, 0.4619397662556433, -0.4157348061512721, 0.3535533905932733,
         -0.2777851165098008, 0.1913417161825431, -0.0975451610080625]
    ]).to(x.device)

    assert norm == 'ortho'

    dct_val = x @ DCT_MAT
    dct_val = dct_val.permute(0, 1, 3, 2) @ DCT_MAT

    return dct_val.permute(0, 1, 3, 2)


def idct_2d_v2(x, norm=None):
    DCT_MAT_INV = torch.Tensor([[0.3535534739494324, 0.3535533845424652, 0.3535533845424652,
             0.3535534143447876, 0.3535534143447876, 0.3535533845424652,
             0.3535533845424652, 0.3535534143447876],
            [0.4903925359249115, 0.4157349169254303, 0.2777851521968842,
             0.0975451469421387, -0.0975451767444611, -0.2777851223945618,
             -0.4157347977161407, -0.4903926253318787],
            [0.4619398415088654, 0.1913416683673859, -0.1913417279720306,
             -0.4619398117065430, -0.4619398117065430, -0.1913417279720306,
             0.1913416683673859, 0.4619397521018982],
            [0.4157347977161407, -0.0975451320409775, -0.4903926849365234,
             -0.2777851223945618, 0.2777851819992065, 0.4903926849365234,
             0.0975451916456223, -0.4157348275184631],
            [0.3535533845424652, -0.3535533845424652, -0.3535533845424652,
             0.3535533845424652, 0.3535533547401428, -0.3535533845424652,
             -0.3535533845424652, 0.3535533547401428],
            [0.2777851223945618, -0.4903927445411682, 0.0975452065467834,
             0.4157349467277527, -0.4157347679138184, -0.0975451916456223,
             0.4903925955295563, -0.2777850627899170],
            [0.1913416385650635, -0.4619397521018982, 0.4619397521018982,
             -0.1913417577743530, -0.1913417875766754, 0.4619397521018982,
             -0.4619397521018982, 0.1913416683673859],
            [0.0975452512502670, -0.2777851819992065, 0.4157348275184631,
             -0.4903926849365234, 0.4903926849365234, -0.4157347679138184,
             0.2777851223945618, -0.0975450947880745]]).to(x.device)

    assert norm == 'ortho'

    idct_val = x.permute(0, 1, 3, 2) @ DCT_MAT_INV
    idct_val = idct_val.permute(0, 1, 3, 2) @ DCT_MAT_INV

    return idct_val




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    bgr_img = np.random.random((24, ))
    # bgr_tensor = torch.tensor(bgr_img)
    #
    # dct_cv2 = cv2.dct(bgr_img)
    #
    # print(dct_cv2)
    #
    # # x = torch.rand((5, 10))
    # #
    # x_fft = dct(bgr_tensor.reshape(1, 1, -1), norm='ortho')
    # print(x_fft)
    #
    #
    # img2 = torch.randn((1, 1, 24, 36))
    # img2 = torch.ones((1, 1, 24, 36))
    #
    # plt.imshow(img2.squeeze(0).squeeze(0))
    # plt.show()
    # f_img = dct_2d(img2)
    # plt.imshow(f_img.squeeze(0).squeeze(0))
    # plt.show()

    # print((img2 - idct_2d(dct_2d(img2))).abs().sum())


    f_img = torch.zeros((1, 1, 8, 8))
    f_img[0, 0, -1, -1] = 2
    plt.imshow(f_img.squeeze(0).squeeze(0))
    plt.show()

    r_img = idct_2d(f_img)
    plt.imshow(r_img.squeeze(0).squeeze(0))
    plt.colorbar()
    plt.show()





