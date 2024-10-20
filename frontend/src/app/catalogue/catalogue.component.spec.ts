import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CatalogueComponent } from './catalogue.component';

describe('CatalogueComponent', () => {
  let component: CatalogueComponent;
  let fixture: ComponentFixture<CatalogueComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [CatalogueComponent]
    });
    fixture = TestBed.createComponent(CatalogueComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
