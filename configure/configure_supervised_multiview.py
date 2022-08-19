def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(
            type='CG',  # other: CV
            view=3,
            seed=8,
            training=dict(
                lr=3.0e-4,
                start_dual_prediction=500,
                batch_size=256,
                epoch=1000,
                alpha=10,
                lambda1=0.1,
                lambda2=0.1,
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],
                arch2=[512, 1024, 1024, 1024, 128],
                arch3=[928, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
            ),
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            type='CG',  # other: CV
            view=3,
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                arch3=[40, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
            ),
            training=dict(
                lr=1.0e-4,
                start_dual_prediction=0,
                batch_size=256,
                epoch=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.1
            ),
            seed=8,
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            type='CG',  # other: CV
            view=3,
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 40],
                arch2=[40, 1024, 1024, 1024, 40],
                arch3=[20, 1024, 1024, 1024, 40],
                activations='relu',
                batchnorm=True,
            ),
            training=dict(
                lr=1.0e-4,
                start_dual_prediction=0,
                batch_size=256,
                epoch=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.1,
            ),
            seed=2,
        )

    else:
        raise Exception('Undefined data name')
