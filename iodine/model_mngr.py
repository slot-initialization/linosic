import torch
from torch import optim
from dataset_mngr import DataSetCreator as DSC
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, BatchSampler
from model.iodine import IODINE


class ModelManagement:
    def __init__(self, options, train_root=None, test_eval_root=None, mask_root=None, eval_checkpoint_path=None, resume_checkpoint_path=None):
        self.opt = options.opt
        self.mode = options.mode
        self.eval_checkpoint_path = eval_checkpoint_path
        self.resume_checkpoint_path = resume_checkpoint_path
        self.device_type = self.opt.device_type
        self.device_ids = self.opt.device_ids
        self.model = self.make_model(options)
        if self.opt.mode == 'train':
            self.optimizer = self.make_optimizer()
            if self.opt.resume_training:
                self.load_state()
            self.train_data_loader, \
                self.sampler = self.make_dataloader(split_root=train_root,
                                                    mask_root='',
                                                    size=self.opt.train_dataset_size,
                                                    batch_size=self.opt.train_batch_size,
                                                    mask=False,
                                                    shuffle=True)
            print('Created train Dataset without masks of size', len(self.train_data_loader))
        self.test_data_loader, \
            self.sampler = self.make_dataloader(split_root=test_eval_root,
                                                mask_root=mask_root,
                                                size=self.opt.test_dataset_size,
                                                batch_size=1,
                                                mask=True,
                                                shuffle=False)
        print('Created test or eval Dataset with masks of size', len(self.test_data_loader))

    def make_model(self, options):
        model = IODINE(options)
        model = model.to(self.device_type)
        #if self.opt.mode == 'eval':
        #    model.load_state_dict(torch.load(self.eval_checkpoint_path)['model'])
        """ If state loading does not work try the loading below"""
        if self.opt.mode == 'eval':
            model = torch.nn.DataParallel(model)
            try:
                model.load_state_dict(torch.load(self.eval_checkpoint_path)['model_state_dict'])
            except:
                pass
            try:
                model.load_state_dict(torch.load(self.eval_checkpoint_path)['model'])
            except:
                pass
            model = model.module
            try:
                model.load_state_dict(torch.load(self.eval_checkpoint_path)['model_state_dict'])
            except:
                pass
            try:
                model.load_state_dict(torch.load(self.eval_checkpoint_path)['model'])
            except:
                pass


        return model

    def make_optimizer(self):
        learning_rate = self.opt.base_learning_rate
        weight_decay = self.opt.weight_decay
        params = []
        for key, value in self.model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": learning_rate, "weight_decay": weight_decay}]
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def make_dataloader(self, split_root, mask_root, size=-1, batch_size=32, mask=False, shuffle=True):
        dataset = DSC(options=self.opt,
                      max_num_slots=self.opt.slots,
                      split_root=split_root,
                      mask_root=mask_root,
                      size=size,
                      resolution=(self.opt.resolution, self.opt.resolution),
                      mask=mask)
        num_workers = self.opt.num_workers
        if self.opt.parallel:
            sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
        else:
            sampler = RandomSampler(dataset) if self.opt.mode == 'train' else SequentialSampler(dataset)
        if self.opt.mode == 'train':
            batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True)
            data_loader = DataLoader(dataset=dataset,
                                     batch_sampler=batch_sampler,
                                     num_workers=num_workers,
                                     pin_memory=True)
            return data_loader, sampler
        else:
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     sampler=sampler,
                                     drop_last=False,
                                     num_workers=num_workers,
                                     pin_memory=True)
        return data_loader, sampler

    def get_state(self):
        state_dict = {'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        return state_dict

    def load_state(self):
        print('resuming model')
        state_dict = torch.load(self.resume_checkpoint_path)
        self.model.load_state_dict(state_dict=state_dict['model'])
        self.optimizer.load_state_dict(state_dict=state_dict['optimizer'])








