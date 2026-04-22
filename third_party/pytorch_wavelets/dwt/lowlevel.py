import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from pytorch_wavelets.utils import reflect
import pywt


def roll(x, n, dim, make_even=False):
    """
    沿指定维度滚动张量
    
    参数:
        x (tensor): 输入张量
        n (int): 滚动步数
        dim (int): 滚动的维度
        make_even (bool): 是否在奇数长度时进行调整
    """
    if n < 0:
        n = x.shape[dim] + n

    if make_even and x.shape[dim] % 2 == 1:
        end = 1
    else:
        end = 0

    if dim == 0:
        return torch.cat((x[-n:], x[:-n+end]), dim=0)
    elif dim == 1:
        return torch.cat((x[:,-n:], x[:,:-n+end]), dim=1)
    elif dim == 2 or dim == -2:
        return torch.cat((x[:,:,-n:], x[:,:,:-n+end]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:,:,:,-n:], x[:,:,:,:-n+end]), dim=3)


def mypad(x, pad, mode='constant', value=0):
    """ 
    对张量进行类似numpy的填充操作。仅适用于2D填充。
    
    输入:
        x (tensor): 要填充的张量
        pad (tuple): (左, 右, 上, 下) 填充大小的元组
        mode (str): 'symmetric', 'wrap', 'constant, 'reflect', 'replicate', 或
            'zero'。填充技术。
    """
    if mode == 'symmetric':
        # 仅垂直方向
        if pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,xe]
        # 仅水平方向
        elif pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,:,xe]
        # 两个方向都填充
        else:
            m1, m2 = pad[0], pad[1]
            l1 = x.shape[-1]
            xe_row = reflect(np.arange(-m1, l1+m2, dtype='int32'), -0.5, l1-0.5)
            m1, m2 = pad[2], pad[3]
            l2 = x.shape[-2]
            xe_col = reflect(np.arange(-m1, l2+m2, dtype='int32'), -0.5, l2-0.5)
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]
    elif mode == 'periodic':
        # 仅垂直方向
        if pad[0] == 0 and pad[1] == 0:
            xe = np.arange(x.shape[-2])
            xe = np.pad(xe, (pad[2], pad[3]), mode='wrap')
            return x[:,:,xe]
        # 仅水平方向
        elif pad[2] == 0 and pad[3] == 0:
            xe = np.arange(x.shape[-1])
            xe = np.pad(xe, (pad[0], pad[1]), mode='wrap')
            return x[:,:,:,xe]
        # 两个方向都填充
        else:
            xe_col = np.arange(x.shape[-2])
            xe_col = np.pad(xe_col, (pad[2], pad[3]), mode='wrap')
            xe_row = np.arange(x.shape[-1])
            xe_row = np.pad(xe_row, (pad[0], pad[1]), mode='wrap')
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]

    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        return F.pad(x, pad)
    else:
        raise ValueError("未知的填充类型: {}".format(mode))


def afb1d(x, h0, h1, use_amp, mode='zero', dim=-1):
    """ 
    图像的一维分析滤波器组（仅沿一个维度）
    
    输入:
        x (tensor): 4D输入，最后两个维度是空间输入
        h0 (tensor): 低通滤波器的4D输入。形状应为 (1, 1, h, 1) 或 (1, 1, 1, w)
        h1 (tensor): 高通滤波器的4D输入。形状应为 (1, 1, h, 1) 或 (1, 1, 1, w)
        mode (str): 填充方法
        dim (int) - 滤波的维度。d=2是垂直滤波器（称为列滤波但跨行滤波）。
            d=3是水平滤波器（称为行滤波但跨列滤波）。

    返回:
        lohi: 沿通道维度连接的低通和高通子带
    """
    C = x.shape[1]
    # 将维度转换为正数
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    # 如果h0, h1不是张量，则创建它们。如果是，则假定它们顺序正确
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1]
    shape[d] = L
    # 如果h的形状不正确，则调整形状
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode == 'per' or mode == 'periodization':
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:,:,-1:]), dim=2)
            else:
                x = torch.cat((x, x[:,:,:,-1:]), dim=3)
            N += 1
        x = roll(x, -L2, dim=d)
        pad = (L-1, 0) if d == 2 else (0, L-1)
        if use_amp:
            with torch.cuda.amp.autocast(): # 混合精度
                lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        else:
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N//2
        if d == 2:
            lohi[:,:,:L2] = lohi[:,:,:L2] + lohi[:,:,N2:N2+L2]
            lohi = lohi[:,:,:N2]
        else:
            lohi[:,:,:,:L2] = lohi[:,:,:,:L2] + lohi[:,:,:,N2:N2+L2]
            lohi = lohi[:,:,:,:N2]
    else:
        # 计算填充大小
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            # 遗憾的是，pytorch只允许前后相同的填充，如果需要对奇数长度信号进行更多后填充，
            # 必须预先填充
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p//2, 0) if d == 2 else (0, p//2)
            # 计算高低通
            if use_amp:
                with torch.cuda.amp.autocast():
                    lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
            else:
                lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (0, 0, p//2, (p+1)//2) if d == 2 else (p//2, (p+1)//2, 0, 0)
            x = mypad(x, pad=pad, mode=mode)
            if use_amp:
                with torch.cuda.amp.autocast():
                    lohi = F.conv2d(x, h, stride=s, groups=C)
            else:
                lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError("未知的填充类型: {}".format(mode))

    return lohi


def afb1d_atrous(x, h0, h1, mode='periodic', dim=-1, dilation=1):
    """ 
    图像的一维分析滤波器组（仅沿一个维度），不进行下采样。执行a trous算法。
    
    输入:
        x (tensor): 4D输入，最后两个维度是空间输入
        h0 (tensor): 低通滤波器的4D输入。形状应为 (1, 1, h, 1) 或 (1, 1, 1, w)
        h1 (tensor): 高通滤波器的4D输入。形状应为 (1, 1, h, 1) 或 (1, 1, 1, w)
        mode (str): 填充方法
        dim (int) - 滤波的维度。d=2是垂直滤波器（称为列滤波但跨行滤波）。
            d=3是水平滤波器（称为行滤波但跨列滤波）。
        dilation (int): 膨胀因子。应为2的幂。

    返回:
        lohi: 沿通道维度连接的低通和高通子带
    """
    C = x.shape[1]
    # 将维度转换为正数
    d = dim % 4
    # 如果h0, h1不是张量，则创建它们。如果是，则假定它们顺序正确
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    # 如果h的形状不正确，则调整形状
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    # 计算填充大小
    L2 = (L * dilation)//2
    pad = (0, 0, L2-dilation, L2) if d == 2 else (L2-dilation, L2, 0, 0)
    x = mypad(x, pad=pad, mode=mode)
    lohi = F.conv2d(x, h, groups=C, dilation=dilation)

    return lohi


def sfb1d(lo, hi, g0, g1, use_amp, mode='zero', dim=-1):
    """ 
    图像张量的一维合成滤波器组
    """
    C = lo.shape[1]
    d = dim % 4
    # 如果g0, g1不是张量，则创建它们。如果是，则假定它们顺序正确
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    N = 2*lo.shape[d]
    # 如果g的形状不正确，则调整形状
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1,2)
    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)
    if mode == 'per' or mode == 'periodization':
        if use_amp:
            with torch.cuda.amp.autocast():
                y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
                    F.conv_transpose2d(hi, g1, stride=s, groups=C)
        else:
            y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
                F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,N:N+L-2]
            y = y[:,:,:N]
        else:
            y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,N:N+L-2]
            y = y[:,:,:,:N]
        y = roll(y, 1-L//2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
                mode == 'periodic':
            pad = (L-2, 0) if d == 2 else (0, L-2)
            if use_amp:
                with torch.cuda.amp.autocast():
                    y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                        F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
            else:
                y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                    F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("未知的填充类型: {}".format(mode))

    return y


def mode_to_int(mode):
    """
    将填充模式字符串转换为整数编码
    """
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("未知的填充类型: {}".format(mode))


def int_to_mode(mode):
    """
    将整数编码转换为填充模式字符串
    """
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("未知的填充类型: {}".format(mode))


class AFB2D(Function):
    """ 
    对输入进行单层2D小波分解。通过两次调用
    :py:func:[pytorch_wavelets.dwt.lowlevel.afb1d](file://c:\\Users\\Administrator\\Desktop\\WPMixer-main\\pytorch_wavelets\\dwt\\lowlevel.py#L90-L183)执行单独的行和列滤波
    
    需要将张量转换为正确的形式。由于该函数定义了自己的反向传播，
    通过不保存输入张量来节省内存。
    
    输入:
        x (torch.Tensor): 要分解的输入
        h0_row: 行低通
        h1_row: 行高通
        h0_col: 列低通
        h1_col: 列高通
        mode (int): 使用mode_to_int获取此处的整数代码

    我们将模式编码为整数而不是字符串，因为gradcheck在提供字符串时会导致错误。

    返回:
        y: 形状为 (N, C*4, H, W) 的张量
    """
    @staticmethod
    def forward(ctx, x, h0_row, h1_row, h0_col, h1_col, mode):
        ctx.save_for_backward(h0_row, h1_row, h0_col, h1_col)
        ctx.shape = x.shape[-2:]
        mode = int_to_mode(mode)
        ctx.mode = mode
        lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
        y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:,:,0].contiguous()
        highs = y[:,:,1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, low, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            lh, hl, hh = torch.unbind(highs, dim=2)
            lo = sfb1d(low, lh, h0_col, h1_col, mode=mode, dim=2)
            hi = sfb1d(hl, hh, h0_col, h1_col, mode=mode, dim=2)
            dx = sfb1d(lo, hi, h0_row, h1_row, mode=mode, dim=3)
            if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:ctx.shape[-2], :ctx.shape[-1]]
            elif dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:,:,:ctx.shape[-2]]
            elif dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:,:ctx.shape[-1]]
        return dx, None, None, None, None, None


class AFB1D(Function):
    """ 
    对输入进行单层1D小波分解。

    需要将张量转换为正确的形式。由于该函数定义了自己的反向传播，
    通过不保存输入张量来节省内存。

    输入:
        x (torch.Tensor): 要分解的输入
        h0: 低通
        h1: 高通
        mode (int): 使用mode_to_int获取此处的整数代码

    我们将模式编码为整数而不是字符串，因为gradcheck在提供字符串时会导致错误。

    返回:
        x0: 形状为 (N, C, L') 的张量 - 低通
        x1: 形状为 (N, C, L') 的张量 - 高通
    """
    @staticmethod
    def forward(ctx, x, h0, h1, mode, use_amp):
        mode = int_to_mode(mode)

        # 将输入转换为4D
        x = x[:, :, None, :]
        h0 = h0[:, :, None, :]
        h1 = h1[:, :, None, :]

        # 保存用于反向传播
        ctx.save_for_backward(h0, h1)
        ctx.shape = x.shape[3]
        ctx.mode = mode
        ctx.use_amp = use_amp
        
        lohi = afb1d(x, h0, h1, use_amp, mode=mode, dim=3)
        x0 = lohi[:, ::2, 0].contiguous()
        x1 = lohi[:, 1::2, 0].contiguous()
        return x0, x1

    @staticmethod
    def backward(ctx, dx0, dx1):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0, h1 = ctx.saved_tensors
            use_amp = ctx.use_amp
            
            # 将梯度转换为4D
            dx0 = dx0[:, :, None, :]
            dx1 = dx1[:, :, None, :]

            dx = sfb1d(dx0, dx1, h0, h1, use_amp, mode=mode, dim=3)[:, :, 0]

            # 检查奇数输入
            if dx.shape[2] > ctx.shape:
                dx = dx[:, :, :ctx.shape]

        return dx, None, None, None, None, None


def afb2d(x, filts, mode='zero'):
    """ 
    对输入进行单层2D小波分解。通过两次调用
    :py:func:[pytorch_wavelets.dwt.lowlevel.afb1d](file://c:\\Users\\Administrator\\Desktop\\WPMixer-main\\pytorch_wavelets\\dwt\\lowlevel.py#L90-L183)执行单独的行和列滤波

    输入:
        x (torch.Tensor): 要分解的输入
        filts (ndarray或torch.Tensor列表): 如果给定了张量列表，
            此函数假定它们是正确的形式（由
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`返回的形式）。
            否则，此函数将通过调用
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`准备正确的滤波器形式。
        mode (str): 'zero', 'symmetric', 'reflect' 或 'periodization'。使用哪种填充。
            如果是periodization，输出大小将是输入大小的一半。
            否则，输出大小将略大于一半。

    返回:
        y: 形状为 (N, C*4, H, W) 的张量
    """
    tensorize = [not isinstance(f, torch.Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                h0, h1, device=x.device)
        else:
            h0_col = h0
            h0_row = h0.transpose(2,3)
            h1_col = h1
            h1_row = h1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                *filts, device=x.device)
        else:
            h0_col, h1_col, h0_row, h1_row = filts
    else:
        raise ValueError("输入滤波器的未知形式")

    lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
    y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)

    return y


def afb2d_atrous(x, filts, mode='periodization', dilation=1):
    """ 
    对输入进行单层2D小波分解。通过两次调用
    :py:func:[pytorch_wavelets.dwt.lowlevel.afb1d](file://c:\\Users\\Administrator\\Desktop\\WPMixer-main\\pytorch_wavelets\\dwt\\lowlevel.py#L90-L183)执行单独的行和列滤波

    输入:
        x (torch.Tensor): 要分解的输入
        filts (ndarray或torch.Tensor列表): 如果给定了张量列表，
            此函数假定它们是正确的形式（由
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`返回的形式）。
            否则，此函数将通过调用
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`准备正确的滤波器形式。
        mode (str): 'zero', 'symmetric', 'reflect' 或 'periodization'。使用哪种填充。
            如果是periodization，输出大小将是输入大小的一半。
            否则，输出大小将略大于一半。
        dilation (int): 滤波器的膨胀因子。应为2**level

    返回:
        y: 形状为 (N, C, 4, H, W) 的张量
    """
    tensorize = [not isinstance(f, torch.Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                h0, h1, device=x.device)
        else:
            h0_col = h0
            h0_row = h0.transpose(2,3)
            h1_col = h1
            h1_row = h1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                *filts, device=x.device)
        else:
            h0_col, h1_col, h0_row, h1_row = filts
    else:
        raise ValueError("输入滤波器的未知形式")

    lohi = afb1d_atrous(x, h0_row, h1_row, mode=mode, dim=3, dilation=dilation)
    y = afb1d_atrous(lohi, h0_col, h1_col, mode=mode, dim=2, dilation=dilation)

    return y


def afb2d_nonsep(x, filts, mode='zero'):
    """ 
    对输入进行1层2D小波分解。不执行单独的行和列滤波。

    输入:
        x (torch.Tensor): 要分解的输入
        filts (list或torch.Tensor): 如果给定列表，应为低通和高通滤波器组。
            如果给定张量，应为由
            :py:func:[pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d_nonsep](file://c:\\Users\\Administrator\\Desktop\\WPMixer-main\\pytorch_wavelets\\dwt\\lowlevel.py#L826-L858)创建的形式
        mode (str): 'zero', 'symmetric', 'reflect' 或 'periodization'。使用哪种填充。
            如果是periodization，输出大小将是输入大小的一半。
            否则，输出大小将略大于一半。

    返回:
        y: 形状为 (N, C, 4, H, W) 的张量
    """
    C = x.shape[1]
    Ny = x.shape[2]
    Nx = x.shape[3]

    # 检查滤波器输入
    if isinstance(filts, (tuple, list)):
        if len(filts) == 2:
            filts = prep_filt_afb2d_nonsep(filts[0], filts[1], device=x.device)
        else:
            filts = prep_filt_afb2d_nonsep(
                filts[0], filts[1], filts[2], filts[3], device=x.device)
    f = torch.cat([filts]*C, dim=0)
    Ly = f.shape[2]
    Lx = f.shape[3]

    if mode == 'periodization' or mode == 'per':
        if x.shape[2] % 2 == 1:
            x = torch.cat((x, x[:,:,-1:]), dim=2)
            Ny += 1
        if x.shape[3] % 2 == 1:
            x = torch.cat((x, x[:,:,:,-1:]), dim=3)
            Nx += 1
        pad = (Ly-1, Lx-1)
        stride = (2, 2)
        x = roll(roll(x, -Ly//2, dim=2), -Lx//2, dim=3)
        y = F.conv2d(x, f, padding=pad, stride=stride, groups=C)
        y[:,:,:Ly//2] += y[:,:,Ny//2:Ny//2+Ly//2]
        y[:,:,:,:Lx//2] += y[:,:,:,Nx//2:Nx//2+Lx//2]
        y = y[:,:,:Ny//2, :Nx//2]
    elif mode == 'zero' or mode == 'symmetric' or mode == 'reflect':
        # 计算填充大小
        out1 = pywt.dwt_coeff_len(Ny, Ly, mode=mode)
        out2 = pywt.dwt_coeff_len(Nx, Lx, mode=mode)
        p1 = 2 * (out1 - 1) - Ny + Ly
        p2 = 2 * (out2 - 1) - Nx + Lx
        if mode == 'zero':
            # 遗憾的是，pytorch只允许前后相同的填充，如果需要对奇数长度信号进行更多后填充，
            # 必须预先填充
            if p1 % 2 == 1 and p2 % 2 == 1:
                x = F.pad(x, (0, 1, 0, 1))
            elif p1 % 2 == 1:
                x = F.pad(x, (0, 0, 0, 1))
            elif p2 % 2 == 1:
                x = F.pad(x, (0, 1, 0, 0))
            # 计算高低通
            y = F.conv2d(
                x, f, padding=(p1//2, p2//2), stride=2, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (p2//2, (p2+1)//2, p1//2, (p1+1)//2)
            x = mypad(x, pad=pad, mode=mode)
            y = F.conv2d(x, f, stride=2, groups=C)
    else:
        raise ValueError("未知的填充类型: {}".format(mode))

    return y


def sfb2d(ll, lh, hl, hh, filts, mode='zero'):
    """ 
    对小波系数进行单层2D小波重构。通过两次调用
    :py:func:[pytorch_wavelets.dwt.lowlevel.sfb1d](file://c:\\Users\\Administrator\\Desktop\\WPMixer-main\\pytorch_wavelets\\dwt\\lowlevel.py#L237-L292)执行单独的行和列滤波

    输入:
        ll (torch.Tensor): 低通系数
        lh (torch.Tensor): 水平系数
        hl (torch.Tensor): 垂直系数
        hh (torch.Tensor): 对角系数
        filts (ndarray或torch.Tensor列表): 如果给定了张量列表，
            此函数假定它们是正确的形式（由
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`返回的形式）。
            否则，此函数将通过调用
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`准备正确的滤波器形式。
        mode (str): 'zero', 'symmetric', 'reflect' 或 'periodization'。使用哪种填充。
            如果是periodization，输出大小将是输入大小的一半。
            否则，输出大小将略大于一半。
    """
    tensorize = [not isinstance(x, torch.Tensor) for x in filts]
    if len(filts) == 2:
        g0, g1 = filts
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(g0, g1)
        else:
            g0_col = g0
            g0_row = g0.transpose(2,3)
            g1_col = g1
            g1_row = g1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(*filts)
        else:
            g0_col, g1_col, g0_row, g1_row = filts
    else:
        raise ValueError("输入滤波器的未知形式")

    lo = sfb1d(ll, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)

    return y


class SFB2D(Function):
    """ 
    对输入进行单层2D小波分解。通过两次调用
    :py:func:[pytorch_wavelets.dwt.lowlevel.afb1d](file://c:\\Users\\Administrator\\Desktop\\WPMixer-main\\pytorch_wavelets\\dwt\\lowlevel.py#L90-L183)执行单独的行和列滤波

    需要将张量转换为正确的形式。由于该函数定义了自己的反向传播，
    通过不保存输入张量来节省内存。

    输入:
        x (torch.Tensor): 要分解的输入
        h0_row: 行低通
        h1_row: 行高通
        h0_col: 列低通
        h1_col: 列高通
        mode (int): 使用mode_to_int获取此处的整数代码

    我们将模式编码为整数而不是字符串，因为gradcheck在提供字符串时会导致错误。

    返回:
        y: 形状为 (N, C*4, H, W) 的张量
    """
    @staticmethod
    def forward(ctx, low, highs, g0_row, g1_row, g0_col, g1_col, mode):
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0_row, g1_row, g0_col, g1_col)

        lh, hl, hh = torch.unbind(highs, dim=2)
        lo = sfb1d(low, lh, g0_col, g1_col, mode=mode, dim=2)
        hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
        y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)
        return y

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0_row, g1_row, g0_col, g1_col = ctx.saved_tensors
            dx = afb1d(dy, g0_row, g1_row, mode=mode, dim=3)
            dx = afb1d(dx, g0_col, g1_col, mode=mode, dim=2)
            s = dx.shape
            dx = dx.reshape(s[0], -1, 4, s[-2], s[-1])
            dlow = dx[:,:,0].contiguous()
            dhigh = dx[:,:,1:].contiguous()
        return dlow, dhigh, None, None, None, None, None


class SFB1D(Function):
    """ 
    对输入进行单层1D小波分解。

    需要将张量转换为正确的形式。由于该函数定义了自己的反向传播，
    通过不保存输入张量来节省内存。

    输入:
        low (torch.Tensor): 要重构的低通，形状为 (N, C, L)
        high (torch.Tensor): 要重构的高通，形状为 (N, C, L)
        g0: 低通
        g1: 高通
        mode (int): 使用mode_to_int获取此处的整数代码

    我们将模式编码为整数而不是字符串，因为gradcheck在提供字符串时会导致错误。

    返回:
        y: 形状为 (N, C*2, L') 的张量
    """
    @staticmethod
    def forward(ctx, low, high, g0, g1, mode, use_amp):
        mode = int_to_mode(mode)
        # 转换为具有1行的2D张量
        low = low[:, :, None, :]
        high = high[:, :, None, :]
        g0 = g0[:, :, None, :]
        g1 = g1[:, :, None, :]

        ctx.mode = mode
        ctx.save_for_backward(g0, g1)
        ctx.use_amp = use_amp
        
        return sfb1d(low, high, g0, g1, use_amp, mode=mode, dim=3)[:, :, 0]

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            use_amp = ctx.use_amp
            g0, g1, = ctx.saved_tensors
            dy = dy[:, :, None, :]

            dx = afb1d(dy, g0, g1, use_amp, mode=mode, dim=3)

            dlow = dx[:, ::2, 0].contiguous()
            dhigh = dx[:, 1::2, 0].contiguous()
        return dlow, dhigh, None, None, None, None, None


def sfb2d_nonsep(coeffs, filts, mode='zero'):
    """ 
    对小波系数进行单层2D小波重构。不执行可分离滤波。

    输入:
        coeffs (torch.Tensor): 系数张量，形状为 (N, C, 4, H, W)
            其中第三维度索引(ll, lh, hl, hh)频带。
        filts (ndarray或torch.Tensor列表): 如果给定了张量列表，
            此函数假定它们是正确的形式（由
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d_nonsep`返回的形式）。
            否则，此函数将通过调用
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d_nonsep`准备正确的滤波器形式。
        mode (str): 'zero', 'symmetric', 'reflect' 或 'periodization'。使用哪种填充。
            如果是periodization，输出大小将是输入大小的一半。
            否则，输出大小将略大于一半。
    """
    C = coeffs.shape[1]
    Ny = coeffs.shape[-2]
    Nx = coeffs.shape[-1]

    # 检查滤波器输入 - 应该是torch张量的形式，但如果不是，则在此处张量化。
    if isinstance(filts, (tuple, list)):
        if len(filts) == 2:
            filts = prep_filt_sfb2d_nonsep(filts[0], filts[1],
                                           device=coeffs.device)
        elif len(filts) == 4:
            filts = prep_filt_sfb2d_nonsep(
                filts[0], filts[1], filts[2], filts[3], device=coeffs.device)
        else:
            raise ValueError("输入滤波器的未知形式")
    f = torch.cat([filts]*C, dim=0)
    Ly = f.shape[2]
    Lx = f.shape[3]

    x = coeffs.reshape(coeffs.shape[0], -1, coeffs.shape[-2], coeffs.shape[-1])
    if mode == 'periodization' or mode == 'per':
        ll = F.conv_transpose2d(x, f, groups=C, stride=2)
        ll[:,:,:Ly-2] += ll[:,:,2*Ny:2*Ny+Ly-2]
        ll[:,:,:,:Lx-2] += ll[:,:,:,2*Nx:2*Nx+Lx-2]
        ll = ll[:,:,:2*Ny,:2*Nx]
        ll = roll(roll(ll, 1-Ly//2, dim=2), 1-Lx//2, dim=3)
    elif mode == 'symmetric' or mode == 'zero' or mode == 'reflect' or \
            mode == 'periodic':
        pad = (Ly-2, Lx-2)
        ll = F.conv_transpose2d(x, f, padding=pad, groups=C, stride=2)
    else:
        raise ValueError("未知的填充类型: {}".format(mode))

    return ll.contiguous()


def prep_filt_afb2d_nonsep(h0_col, h1_col, h0_row=None, h1_row=None,
                           device=None):
    """
    准备用于afb2d_nonsep函数的正确形式的滤波器。
    特别是制作2D点扩散函数，并镜像它们以准备执行torch.conv2d。

    输入:
        h0_col (array-like): 低通列滤波器组
        h1_col (array-like): 高通列滤波器组
        h0_row (array-like): 低通行滤波器组。如果为None，将假定与列滤波器相同
        h1_row (array-like): 高通行滤波器组。如果为None，将假定与列滤波器相同
        device: 将张量放在哪个设备上

    返回:
        filts: (4, 1, h, w) 张量，准备好获取四个子带
    """
    h0_col = np.array(h0_col).ravel()
    h1_col = np.array(h1_col).ravel()
    if h0_row is None:
        h0_row = h0_col
    if h1_row is None:
        h1_row = h1_col
    ll = np.outer(h0_col, h0_row)
    lh = np.outer(h1_col, h0_row)
    hl = np.outer(h0_col, h1_row)
    hh = np.outer(h1_col, h1_row)
    filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                      hl[None,::-1,::-1], hh[None,::-1,::-1]], axis=0)
    filts = torch.tensor(filts, dtype=torch.get_default_dtype(), device=device)
    return filts


def prep_filt_sfb2d_nonsep(g0_col, g1_col, g0_row=None, g1_row=None,
                           device=None):
    """
    准备用于sfb2d_nonsep函数的正确形式的滤波器。
    特别是制作2D点扩散函数。不镜像它们，因为sfb2d_nonsep使用conv2d_transpose，
    它像正常卷积一样工作。

    输入:
        g0_col (array-like): 低通列滤波器组
        g1_col (array-like): 高通列滤波器组
        g0_row (array-like): 低通行滤波器组。如果为None，将假定与列滤波器相同
        g1_row (array-like): 高通行滤波器组。如果为None，将假定与列滤波器相同
        device: 将张量放在哪个设备上

    返回:
        filts: (4, 1, h, w) 张量，准备好组合四个子带
    """
    g0_col = np.array(g0_col).ravel()
    g1_col = np.array(g1_col).ravel()
    if g0_row is None:
        g0_row = g0_col
    if g1_row is None:
        g1_row = g1_col
    ll = np.outer(g0_col, g0_row)
    lh = np.outer(g1_col, g0_row)
    hl = np.outer(g0_col, g1_row)
    hh = np.outer(g1_col, g1_row)
    filts = np.stack([ll[None], lh[None], hl[None], hh[None]], axis=0)
    filts = torch.tensor(filts, dtype=torch.get_default_dtype(), device=device)
    return filts


def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    """
    准备用于sfb2d函数的正确形式的滤波器。特别地，使张量具有正确的形状。
    不镜像它们，因为sfb2d使用conv2d_transpose，它像正常卷积一样工作。

    输入:
        g0_col (array-like): 低通列滤波器组
        g1_col (array-like): 高通列滤波器组
        g0_row (array-like): 低通行滤波器组。如果为None，将假定与列滤波器相同
        g1_row (array-like): 高通行滤波器组。如果为None，将假定与列滤波器相同
        device: 将张量放在哪个设备上

    返回:
        (g0_col, g1_col, g0_row, g1_row)
    """
    g0_col, g1_col = prep_filt_sfb1d(g0_col, g1_col, device)
    if g0_row is None:
        g0_row, g1_row = g0_col, g1_col
    else:
        g0_row, g1_row = prep_filt_sfb1d(g0_row, g1_row, device)

    g0_col = g0_col.reshape((1, 1, -1, 1))
    g1_col = g1_col.reshape((1, 1, -1, 1))
    g0_row = g0_row.reshape((1, 1, 1, -1))
    g1_row = g1_row.reshape((1, 1, 1, -1))

    return g0_col, g1_col, g0_row, g1_row


def prep_filt_sfb1d(g0, g1, device=None):
    """
    准备用于sfb1d函数的正确形式的滤波器。特别地，使张量具有正确的形状。
    不镜像它们，因为sfb2d使用conv2d_transpose，它像正常卷积一样工作。

    输入:
        g0 (array-like): 低通滤波器组
        g1 (array-like): 高通滤波器组
        device: 将张量放在哪个设备上

    返回:
        (g0, g1)
    """
    g0 = np.array(g0).ravel()
    g1 = np.array(g1).ravel()
    t = torch.get_default_dtype()
    g0 = torch.tensor(g0, device=device, dtype=t).reshape((1, 1, -1))
    g1 = torch.tensor(g1, device=device, dtype=t).reshape((1, 1, -1))

    return g0, g1


def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
    """
    准备用于afb2d函数的正确形式的滤波器。特别地，使张量具有正确的形状。
    镜像它们，因为afb2d使用conv2d，它像正常相关一样工作。

    输入:
        h0_col (array-like): 低通列滤波器组
        h1_col (array-like): 高通列滤波器组
        h0_row (array-like): 低通行滤波器组。如果为None，将假定与列滤波器相同
        h1_row (array-like): 高通行滤波器组。如果为None，将假定与列滤波器相同
        device: 将张量放在哪个设备上

    返回:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    if h0_row is None:
        h0_row, h1_row = h0_col, h1_col
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)

    h0_col = h0_col.reshape((1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, -1))
    return h0_col, h1_col, h0_row, h1_row


def prep_filt_afb1d(h0, h1, device=None):
    """
    准备用于afb2d函数的正确形式的滤波器。特别地，使张量具有正确的形状。
    镜像它们，因为afb2d使用conv2d，它像正常相关一样工作。

    输入:
        h0 (array-like): 低通列滤波器组
        h1 (array-like): 高通列滤波器组
        device: 将张量放在哪个设备上

    返回:
        (h0, h1)
    """
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1