import torch.nn as nn



class MultiInputGridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96), n_row=3, n_col=6):
        super(MultiInputGridNet, self).__init__()

        self.n_row = n_row
        self.n_col = n_col
        self.n_chs = grid_chs
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'
        assert len(in_chs) == self.n_row, 'should give input channels for each row (scale stream)'

        for r, n_ch in enumerate(self.n_chs):
            setattr(self, f'lateral_{r}_0', LateralBlock(in_chs[r], n_ch))
            for c in range(1, self.n_col):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, *args):
        assert len(args) == self.n_row

        # extensible, memory-efficient
        cur_col = list(args)
        for c in range(int(self.n_col/2)):
            for r in range(self.n_row):
                cur_col[r] = getattr(self, f'lateral_{r}_{c}')(cur_col[r])
                if r != 0:
                    cur_col[r] += getattr(self, f'down_{r-1}_{c}')(cur_col[r-1])
        
        for c in range(int(self.n_col/2), self.n_col):
            for r in range(self.n_row-1, -1, -1):
                cur_col[r] = getattr(self, f'lateral_{r}_{c}')(cur_col[r])
                if r != self.n_row-1:
                    cur_col[r] += getattr(self, f'up_{r}_{c-int(self.n_col/2)}')(cur_col[r+1])

        return self.lateral_final(cur_col[0])

        # extensible, brute-force
        # states = {}
        # for c in range(int(self.n_col/2)):
        #     for r in range(self.n_row):
        #         if c == 0:
        #             states[f'{r}{c}'] = getattr(self, f'lateral_{r}_{c}')(args[r])
        #         else:
        #             states[f'{r}{c}'] = getattr(self, f'lateral_{r}_{c}')(states[f'{r}{c-1}'])
        #         if r != 0:
        #             states[f'{r}{c}'] += getattr(self, f'down_{r-1}_{c}')(states[f'{r-1}{c}'])
        
        # for c in range(int(self.n_col/2), self.n_col):
        #     for r in range(self.n_row-1, -1, -1):
        #         states[f'{r}{c}'] = getattr(self, f'lateral_{r}_{c}')(states[f'{r}{c-1}'])
        #         if r != 2:
        #             states[f'{r}{c}'] += getattr(self, f'up{r}_{c-3}')(states[f'{r+1}{c}'])


        # memory-efficient, non-extensible
        # state_00 = self.lateral_0_0(args[0])
        # state_10 = self.down_0_0(state_00) + self.lateral_1_0(args[1])
        # state_20 = self.down_1_0(state_10) + self.lateral_2_0(args[2])

        # state_01 = self.lateral_0_1(state_00)
        # state_11 = self.down_0_1(state_01) + self.lateral_1_1(state_10)
        # state_21 = self.down_1_1(state_11) + self.lateral_2_1(state_20)

        # state_02 = self.lateral_0_2(state_01)
        # state_12 = self.down_0_2(state_02) + self.lateral_1_2(state_11)
        # state_22 = self.down_1_2(state_12) + self.lateral_2_2(state_21)

        # state_23 = self.lateral_2_3(state_22)
        # state_13 = self.up_1_0(state_23) + self.lateral_1_3(state_12)
        # state_03 = self.up_0_0(state_13) + self.lateral_0_3(state_02)

        # state_24 = self.lateral_2_4(state_23)
        # state_14 = self.up_1_1(state_24) + self.lateral_1_4(state_13)
        # state_04 = self.up_0_1(state_14) + self.lateral_0_4(state_03)

        # state_25 = self.lateral_2_5(state_24)
        # state_15 = self.up_1_2(state_25) + self.lateral_1_5(state_14)
        # state_05 = self.up_0_2(state_15) + self.lateral_0_5(state_04)

        # return self.lateral_final(state_05)


class MIMOGridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96), n_row=3, n_col=6, outrow=(0,1,2)):
        super(MIMOGridNet, self).__init__()

        self.n_row = n_row
        self.n_col = n_col
        self.n_chs = grid_chs
        self.outrow = outrow
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'
        assert len(in_chs) == self.n_row, 'should give input channels for each row (scale stream)'
        assert len(out_chs) == len(self.outrow), 'should give out channels for each output row (scale stream)'

        for r, n_ch in enumerate(self.n_chs):
            setattr(self, f'lateral_{r}_0', LateralBlock(in_chs[r], n_ch))
            for c in range(1, self.n_col):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        for i, r in enumerate(outrow):
            setattr(self, f'lateral_final_{r}', LateralBlock(self.n_chs[r], out_chs[i]))


    def forward(self, *args):
        assert len(args) == self.n_row

        # extensible, memory-efficient
        cur_col = list(args)
        for c in range(int(self.n_col/2)):
            for r in range(self.n_row):
                cur_col[r] = getattr(self, f'lateral_{r}_{c}')(cur_col[r])
                if r != 0:
                    cur_col[r] += getattr(self, f'down_{r-1}_{c}')(cur_col[r-1])
        
        for c in range(int(self.n_col/2), self.n_col):
            for r in range(self.n_row-1, -1, -1):
                cur_col[r] = getattr(self, f'lateral_{r}_{c}')(cur_col[r])
                if r != self.n_row-1:
                    cur_col[r] += getattr(self, f'up_{r}_{c-int(self.n_col/2)}')(cur_col[r+1])

        out = []
        for r in self.outrow:
            out.append(getattr(self, f'lateral_final_{r}')(cur_col[r]))

        return out


class GeneralGridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96), n_row=3, n_col=6):
        super(GeneralGridNet, self).__init__()

        self.n_row = n_row
        self.n_col = n_col
        self.n_chs = grid_chs
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'

        for r, n_ch in enumerate(self.n_chs):
            if r == 0:
                setattr(self, f'lateral_{r}_0', LateralBlock(in_chs, n_ch))
            for c in range(1, self.n_col):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x):
        cur_col = [x] + [None]*(self.n_row-1)
        for c in range(int(self.n_col/2)):
            for r in range(self.n_row):
                if cur_col[r] != None:
                    cur_col[r] = getattr(self, f'lateral_{r}_{c}')(cur_col[r])
                else:
                    cur_col[r] = 0.
                if r != 0:
                    cur_col[r] += getattr(self, f'down_{r-1}_{c}')(cur_col[r-1])
        
        for c in range(int(self.n_col/2), self.n_col):
            for r in range(self.n_row-1, -1, -1):
                cur_col[r] = getattr(self, f'lateral_{r}_{c}')(cur_col[r])
                if r != self.n_row-1:
                    cur_col[r] += getattr(self, f'up_{r}_{c-int(self.n_col/2)}')(cur_col[r+1])

        return self.lateral_final(cur_col[0])


class GridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96)):
        super(GridNet, self).__init__()

        self.n_row = 3
        self.n_col = 6
        self.n_chs = grid_chs
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'

        self.lateral_init = LateralBlock(in_chs, self.n_chs[0])

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col - 1):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)

        return self.lateral_final(state_05)

class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(LateralBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
        return fx + x


class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DownSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.f(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.f(x)