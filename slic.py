import math
from skimage import io, color
import numpy as np
import copy
# from tqdm import trange


class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.side_now = []
        self.side_next = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):

    def __init__(self, images_bbox, K, M):
        self.K = K
        self.M = M

        self.data = images_bbox
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))
        # self.S = int(S)  ## 超像素块大小（边长）

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    @staticmethod
    def open_image(path):
        """
        Return:
            3D array, row col [LAB]
        """
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        :param path:
        :param lab_arr:
        :return:
        """
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def init_clusters(self):
        h = int(self.S / 2)
        w = int(self.S / 2)
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = int(self.S / 2)
            h += int(self.S)

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    # Dc = math.sqrt(
                    #     math.pow(L - cluster.l, 2) +
                    #     math.pow(A - cluster.a, 2) +
                    #     math.pow(B - cluster.b, 2))
                    # Ds = math.sqrt(
                    #     math.pow(h - cluster.h, 2) +
                    #     math.pow(w - cluster.w, 2))
                    # D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    Dc = (math.fabs(L - cluster.l) + math.fabs(A - cluster.a) + math.fabs(B - cluster.b))
                    Ds = (math.fabs(h - cluster.h) + math.fabs(w - cluster.w))
                    D = ((Dc / self.M) + (Ds / self.S))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def noise_pixels_cluster(self, pixels):
        """该聚类中不满足“闭区间”的像素,就 近 原则进行重新分类"""
        cluster_candidate = []
        for cluster in self.clusters:
            if (cluster.h < pixels[0] + 2*self.S) and (cluster.h > pixels[0] - 2*self.S):
                if (cluster.w < pixels[1] + 2*self.S) and (cluster.w > pixels[1] - 2*self.S):
                    cluster_candidate.append(cluster)
        min_distance = 4*self.S
        for candidate in cluster_candidate:
            distance = math.fabs(pixels[0] - candidate.h) + math.fabs(pixels[1] - candidate.w)
            if distance < min_distance:
                cluster_candidate_min = candidate
                min_distance = distance
        self.label[(pixels[0], pixels[1])].pixels.remove((pixels[0], pixels[1]))  ##???在干嘛呢  既有所属，因此先退出之前所属，再加入新的所属
        self.label[(pixels[0], pixels[1])] = cluster_candidate_min
        cluster_candidate_min.pixels.append(pixels)

    """以聚类中心为起始点在满足闭区间的条件下在该聚类像素增长，目的是为了去除该聚类中不满足“闭区间”的像素。"""
    def delete_cluster_noise(self):
        for cluster in self.clusters:
            cluster.side_now.append((cluster.h, cluster.w))
            cluster_pixels_rest = copy.deepcopy(cluster.pixels)
            while len(cluster.side_now) > 0:
                for pixels in cluster.side_now:
                    for i, j in (-1, 0), (1, 0), (0, -1), (0, 1):
                        if (pixels[0]+i, pixels[1]+j) in cluster_pixels_rest:
                            cluster.side_next.append((pixels[0]+i, pixels[1]+j))
                            cluster_pixels_rest.remove((pixels[0]+i, pixels[1]+j))
                            # status = 1
                cluster.side_now = copy.deepcopy(cluster.side_next)
                cluster.side_next = copy.deepcopy([])
            for pixels in cluster_pixels_rest:
                cluster.pixels.remove(pixels)   ## 2 pick 1
                # self.noise_pixels_cluster(pixels)

    def delete_cluster_noise_error(self):
        for cluster in self.clusters:
            cluster.side_now.append((cluster.h, cluster.w))
            cluster_rest = copy.deepcopy(cluster.pixels)
            for pixels in cluster.side_now:
                status = 0
                for i, j in (-1, 0), (1, 0), (0, -1), (0, 1):
                     if (pixels[0]+i, pixels[1]+j) in cluster_rest:
                          cluster.side_now.append((pixels[0]+i, pixels[1]+j))
                          cluster_rest.remove((pixels[0]+i, pixels[1]+j))
                          status = 1
                if status == 0:
                    for pixels in cluster_rest:
                        cluster.pixels.remove(pixels)
                    break
                else:
                    cluster.side_now.remove((pixels[0], pixels[1]))

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)

    def iterate_10times(self):
        self.init_clusters()
        self.move_clusters()
        for i in range(10):
            self.assignment()
            self.update_cluster()
            name = 'lenna_M{m}_K{k}_loop{loop}.png'.format(loop=i, m=self.M, k=self.K)
            self.save_current_image(name)

    def iterate_2times(self):
        self.init_clusters()
        self.move_clusters()
        for i in range(2):
            self.assignment()
            self.update_cluster()
            name = 'lenna_M{m}_K{k}_loop{loop}.png'.format(loop=i, m=self.M, k=self.K)
            self.save_current_image(name)

    def iterate_5times_return_clusters(self):
        self.init_clusters()
        self.move_clusters()
        for i in range(5):
            self.assignment()
            self.update_cluster()
        return self

    def iterate_10times_return_clusters(self):
        self.init_clusters()
        self.move_clusters()
        for i in range(10):
            self.assignment()
            self.update_cluster()
        self.delete_cluster_noise()
        return self


# if __name__ == '__main__':
    # p = SLICProcessor('Lenna.png', 200, 40)
    # p.iterate_10times()
    # p = SLICProcessor('Lenna.png', 300, 40)
    # p.iterate_10times()
    # p = SLICProcessor('Lenna.png', 500, 40)
    # p.iterate_10times()
    # p = SLICProcessor('5.jpg', 4000, 40)
    # p.iterate_2times()
    # p = SLICProcessor('Lenna.png', 200, 5)
    # p.iterate_10times()
    # p = SLICProcessor('Lenna.png', 300, 5)
    # p.iterate_10times()
    # p = SLICProcessor('Lenna.png', 500, 5)
    # p.iterate_10times()
    # p = SLICProcessor('Lenna.png', 1000, 5)
    # p.iterate_10times()

