# play around with the parameters
    blobs = BoxLaplacian(gray, 1.5, 0.05, 1.1, 8)
    interest_points_visualization(up, blobs, None)

    cells = cv2.imread("cv23_lab1_part12_material/cells.jpg")
    cells = cv2.cvtColor(cells, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/cells.jpg", cv2.IMREAD_GRAYSCALE)

    # play around with the parameters
    gray = gray.astype(np.float64)/gray.max()
    blobs = BoxFilters(gray, 4.5, 0.55)
    interest_points_visualization(cells, blobs, None)
    # play around with the parameters
    blobs = BoxLaplacian(gray, 4, 0.35, 1.1, 5)
    interest_points_visualization(cells, blobs, None)


for ix in range(height, x):
    for iy in range(height, y):
        tl1x = ii[ix - midh, iy - midw - width]
        tr1x = ii[ix - midh, iy - midw]
        br1x = ii[ix + midh, iy - midw]
        bl1x = ii[ix + midh, iy - midw - width]

        tl2x = ii[ix - midh, iy - midw]
        tr2x = ii[ix - midh, iy + midw]
        br2x = ii[ix + midh, iy + midw]
        bl2x = ii[ix + midh, iy - midw]

        tl3x = ii[ix - midh, iy + midw]
        tr3x = ii[ix - midh, iy + midw + width]
        br3x = ii[ix + midh, iy + midw + width]
        bl3x = ii[ix + midh, iy + midw]

        height, width = width, height
        midh, midw = int((height - 1)/2), int((width - 1)/2)

        tl1y = ii[ix - midh - height, iy - midw]
        tr1y = ii[ix - midh - height, iy + midw]
        br1y = ii[ix - midh, iy + midw]
        bl1y = ii[ix - midh, iy - midw]

        tl2y = ii[ix - midh, iy - midw]
        tr2y = ii[ix - midh, iy + midw]
        br2y = ii[ix + midh, iy + midw]
        bl2y = ii[ix + midh, iy - midw]

        tl3y = ii[ix + midh, iy - midw]
        tr3y = ii[ix + midh, iy + midw]
        br3y = ii[ix + midh + height, iy + midw]
        bl3y = ii[ix + midh + height, iy - midw]

        height, width = width, height
        midh, midw = int((height - 1)/2), int((width - 1)/2)

        widxy = heixy = width

        tl1xy = ii[ix - 1 - heixy, iy - 1 - widxy]
        tr1xy = ii[ix - 1 - heixy, iy - 1]
        br1xy = ii[ix - 1, iy - 1]
        bl1xy = ii[ix - 1, iy - 1 - widxy]

        tl2xy = ii[ix - 1 - heixy, iy + 1]
        tr2xy = ii[ix - 1 - heixy, iy + 1 + widxy]
        br2xy = ii[ix - 1, iy + 1 + widxy]
        bl2xy = ii[ix - 1, iy + 1]

        tl3xy = ii[ix + 1, iy + 1]
        tr3xy = ii[ix + 1, iy + 1 + widxy]
        br3xy = ii[ix + 1 + heixy, iy + 1 + widxy]
        bl3xy = ii[ix + 1 + heixy, iy + 1]

        tl4xy = ii[ix + 1, iy - 1 - widxy]
        tr4xy = ii[ix + 1, iy - 1]
        br4xy = ii[ix + 1 + heixy, iy - 1]
        bl4xy = ii[ix + 1 + heixy, iy - 1 - widxy]
        
        lxx[ix,iy] = (tl1x - tr1x + br1x - bl1x) - 2*(tl2x - tr2x + br2x - bl2x) + (tl3x - tr3x + br3x - bl3x)
        lyy[ix,iy] = (tl1y - tr1y + br1y - bl1y) - 2*(tl2y - tr2y + br2y - bl2y) + (tl3y - tr3y + br3y - bl3y)
        lxy[ix,iy] = (tl1xy - tr1xy + br1xy - bl1xy) - (tl2xy - tr2xy + br2xy - bl2xy) + (tl3xy - tr3xy + br3xy - bl3xy) - (tl4xy - tr4xy + br4xy - bl4xy)
