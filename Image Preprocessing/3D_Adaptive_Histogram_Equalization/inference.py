import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage import exposure, util
from skimage.data import cells3d

class ImageProcessor:
    def __init__(self, data):
        self.data = data
        self.kernel_size = None
        self.clip_limit = None
        self.sigmoid = None

    def preprocess(self):
        self.data = util.img_as_float(cells3d()[:, 1, :, :])  # grab just the nuclei
        self.data = self.data.transpose()  # Reorder axis order from (z, y, x) to (x, y, z)
        self.data = np.clip(self.data, np.percentile(self.data, 5), np.percentile(self.data, 95))
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

    def degrade_image(self):
        self.sigmoid = np.exp(-3 * np.linspace(0, 1, self.data.shape[0]))
        return (self.data.T * self.sigmoid).T

    def apply_histogram_equalization(self, img):
        return exposure.equalize_hist(img)

    def apply_adaptive_histogram_equalization(self, img, kernel_size, clip_limit):
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        return exposure.equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit)


class VolumeRenderer:
    def __init__(self, data, cmap='Blues'):
        self.data = data
        self.cmap = cmap

    @staticmethod
    def scalars_to_rgba(scalars, cmap, vmin=0.0, vmax=1.0, alpha=0.2):
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgbas = scalar_map.to_rgba(scalars)
        rgbas[:, 3] = alpha
        return rgbas

    def render_volume(self, vol, fig_ax, vmin=0, vmax=1, bin_widths=None, n_levels=20):
        vol = np.clip(vol, vmin, vmax)

        xs, ys, zs = np.mgrid[
            0:vol.shape[0]:bin_widths[0],
            0:vol.shape[1]:bin_widths[1],
            0:vol.shape[2]:bin_widths[2],
        ]
        vol_scaled = vol[::bin_widths[0], ::bin_widths[1], ::bin_widths[2]].flatten()

        levels = np.linspace(vmin, vmax, n_levels)
        alphas = np.linspace(0, 0.7, n_levels)
        alphas = alphas**11
        alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
        alphas *= 0.8

        for il in range(1, len(levels)):
            sel = vol_scaled >= levels[il - 1]
            sel *= vol_scaled <= levels[il]
            if not np.max(sel):
                continue
            c = self.scalars_to_rgba(vol_scaled[sel], self.cmap, vmin=vmin, vmax=vmax, alpha=alphas[il - 1])
            fig_ax.scatter(
                xs.flatten()[sel],
                ys.flatten()[sel],
                zs.flatten()[sel],
                c=c,
                s=0.5 * np.mean(bin_widths),
                marker='o',
                linewidth=0,
            )


class Visualizer:
    def __init__(self, im_orig, im_degraded, im_orig_he, im_degraded_he, im_orig_ahe, im_degraded_ahe, kernel_size):
        self.images = [im_orig, im_orig_he, im_orig_ahe, im_degraded, im_degraded_he, im_degraded_ahe]
        self.kernel_size = kernel_size
        self.cmap = 'Blues'
        self.fig = plt.figure(figsize=(10, 6))
        self.axs = [self.fig.add_subplot(2, 3, i + 1, projection=Axes3D.name, facecolor="none") for i in range(6)]
        self.renderer = VolumeRenderer(self.images, self.cmap)
        self.sigmoid = None
        self.rect_ax = None

    def plot_boxes(self):
        verts = np.array([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]).astype(np.float32)
        lines = [np.array([i, j]) for i in verts for j in verts if np.allclose(np.linalg.norm(i - j), 1)]

        for iax, ax in enumerate(self.axs[:]):
            self.renderer.render_volume(self.images[iax], ax, 0, 1, [2, 2, 2], 20)

            rect_shape = np.array(self.images[0].shape) + 2
            for line in lines:
                ax.plot(
                    (line * rect_shape)[:, 0] - 1,
                    (line * rect_shape)[:, 1] - 1,
                    (line * rect_shape)[:, 2] - 1,
                    linewidth=1,
                    color='gray',
                )

        ns = np.array(self.images[0].shape) // self.kernel_size - 1
        for axis_ind, vertex_ind, box_shape in zip(
            [1] + [2] * 4,
            [[0, 0, 0], [ns[0] - 1, ns[1], ns[2] - 1], [ns[0], ns[1] - 1, ns[2] - 1], [ns[0], ns[1], ns[2] - 1], [ns[0], ns[1], ns[2]]],
            [np.array(self.images[0].shape)] + [self.kernel_size] * 4,
        ):
            for line in lines:
                self.axs[axis_ind].plot(
                    ((line + vertex_ind) * box_shape)[:, 0],
                    ((line + vertex_ind) * box_shape)[:, 1],
                    ((line + vertex_ind) * box_shape)[:, 2],
                    linewidth=1.2,
                    color='crimson',
                )

    def plot_degradation_function(self):
        self.sigmoid = np.exp(-3 * np.linspace(0, 1, self.images[0].shape[0]))
        self.axs[3].scatter(
            xs=np.arange(len(self.sigmoid)),
            ys=np.zeros(len(self.sigmoid)) + self.images[0].shape[1],
            zs=self.sigmoid * self.images[0].shape[2],
            s=5,
            c=self.renderer.scalars_to_rgba(self.sigmoid, cmap=self.cmap, vmin=0, vmax=1, alpha=1.0)[:, :3],
        )

    def customize_subplots(self):
        for iax, ax in enumerate(self.axs[:]):
            for dim_ax in [ax.xaxis, ax.yaxis, ax.zaxis]:
                dim_ax.set_pane_color((1.0, 1.0, 1.0, 0.0))
                dim_ax.line.set_color((1.0, 1.0, 1.0, 0.0))

            xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
            XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
            ax.set_xlim3d(XYZlim)
            ax.set_ylim3d(XYZlim)
            ax.set_zlim3d(XYZlim * 0.5)

            try:
                ax.set_aspect('equal')
            except NotImplementedError:
                pass

            ax.set_xlabel('x', labelpad=-20)
            ax.set_ylabel('y', labelpad=-20)
            ax.text2D(0.63, 0.2, "z", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)
            ax.elev = 30

    def add_annotations(self):
        self.rect_ax = self.fig.add_axes([0, 0, 1, 1], facecolor='none')
        self.rect_ax.set_axis_off()
        rect = patches.Rectangle(
            (0.68, 0.01),
            0.315,
            0.98,
            edgecolor='gray',
            facecolor='none',
            linewidth=2,
            linestyle='--',
        )
        self.rect_ax.add_patch(rect)

        self.rect_ax.text(
            0.19,
            0.34,
            '$I_{degr}(x,y,z) = e^{-x}I_{orig}(x,y,z)$',
            fontsize=9,
            rotation=-15,
            color=self.renderer.scalars_to_rgba([0.8], cmap='Blues', alpha=1.0)[0],
        )

        fc = {'size': 14}
        self.rect_ax.text(
            0.03,
            0.58,
            r'$\it{Original}$' + '\ninput image',
            rotation=90,
            fontdict=fc,
            horizontalalignment='center',
        )
        self.rect_ax.text(
            0.03,
            0.16,
            r'$\it{Degraded}$' + '\ninput image',
            rotation=90,
            fontdict=fc,
            horizontalalignment='center',
        )
        self.rect_ax.text(0.13, 0.91, 'Input volume:\n3D cell image', fontdict=fc)
        self.rect_ax.text(
            0.51,
            0.91,
            r'$\it{Global}$' + '\nhistogram equalization',
            fontdict=fc,
            horizontalalignment='center',
        )
        self.rect_ax.text(
            0.84,
            0.91,
            r'$\it{Adaptive}$' + '\nhistogram equalization (AHE)',
            fontdict=fc,
            horizontalalignment='center',
        )
        self.rect_ax.text(0.58, 0.82, 'non-local', fontsize=12, color='crimson')
        self.rect_ax.text(0.87, 0.82, 'local kernel', fontsize=12, color='crimson')

        cbar_ax = self.fig.add_axes([0.12, 0.43, 0.008, 0.08])
        cbar_ax.imshow(np.arange(256).reshape(256, 1)[::-1], cmap=self.cmap, aspect="auto")
        cbar_ax.set_xticks([])
        cbar_ax.set_yticks([0, 255])
        cbar_ax.set_xticklabels([])
        cbar_ax.set_yticklabels([1.0, 0.0])

    def visualize(self):
        self.plot_boxes()
        self.plot_degradation_function()
        self.customize_subplots()
        self.add_annotations()
        plt.subplots_adjust(
            left=0.05, bottom=-0.1, right=1.01, top=1.1, wspace=-0.1, hspace=-0.45
        )
        plt.show()


# Main execution
def main():
    # Load and process images
    processor = ImageProcessor(cells3d())
    processor.preprocess()
    im_degraded = processor.degrade_image()
    
    im_orig_he = processor.apply_histogram_equalization(processor.data)
    im_degraded_he = processor.apply_histogram_equalization(im_degraded)

    kernel_size = (processor.data.shape[0] // 5, processor.data.shape[1] // 5, processor.data.shape[2] // 2)
    clip_limit = 0.9
    
    im_orig_ahe = processor.apply_adaptive_histogram_equalization(processor.data, kernel_size, clip_limit)
    im_degraded_ahe = processor.apply_adaptive_histogram_equalization(im_degraded, kernel_size, clip_limit)
    
    # Visualize results
    visualizer = Visualizer(processor.data, im_degraded, im_orig_he, im_degraded_he, im_orig_ahe, im_degraded_ahe, kernel_size)
    visualizer.visualize()

if __name__ == '__main__':
    main()

